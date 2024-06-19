import pandas as pd
import numpy as np

def _find_first_transactions(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
):
    """
    Return dataframe with first transactions.

    This takes a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a column named 'repeated' to the transaction log which indicates which rows
    are repeated transactions for that customer_id.

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the column in transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    observation_period_end: :obj: datetime
        a string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    freq: string, optional
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    """

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if type(observation_period_end) == pd.Period:
        observation_period_end = observation_period_end.to_timestamp()

    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].sort_values(select_columns).copy()

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.set_index(datetime_col).to_period(freq).to_timestamp()

    transactions = transactions.loc[(transactions.index <= observation_period_end)].reset_index()

    period_groupby = transactions.groupby([datetime_col, customer_id_col], sort=False, as_index=False)

    if monetary_value_col:
        # when we have a monetary column, make sure to sum together any values in the same period
        period_transactions = period_groupby.sum()
    else:
        # by calling head() on the groupby object, the datetime_col and customer_id_col columns
        # will be reduced
        period_transactions = period_groupby.head(1)

    # initialize a new column where we will indicate which are the first transactions
    period_transactions["first"] = False
    # find all of the initial transactions and store as an index
    first_transactions = period_transactions.groupby(customer_id_col, sort=True, as_index=False).head(1).index
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, "first"] = True
    select_columns.append("first")
    # reset datetime_col to period
    period_transactions[datetime_col] = pd.Index(period_transactions[datetime_col]).to_period(freq)

    return period_transactions[select_columns]


def _find_first_transactions_season(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    high_season_col=None,  # New parameter for high season indicator
    datetime_format=None,
    observation_period_end=None,
    freq="D",
):
    """
    Return dataframe with first transactions and count of high season repeated transactions.

    Parameters
    ----------
    transactions: :obj:`DataFrame`
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id.
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the column in transactions that denotes the monetary value of the transaction.
    high_season_col: string, optional
        the column in transactions that denotes whether a transaction occurred in high season.
    observation_period_end: :obj:`datetime`
        a string or datetime to denote the final date of the study.
    datetime_format: string, optional
        a string that represents the timestamp format.
    freq: string, optional
        'D' for days, or other numpy datetime64 time units.
    """

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if type(observation_period_end) == pd.Period:
        observation_period_end = observation_period_end.to_timestamp()

    select_columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        select_columns.append(monetary_value_col)
    if high_season_col:
        select_columns.append(high_season_col)

    transactions = transactions[select_columns].sort_values([datetime_col, customer_id_col]).copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.loc[transactions[datetime_col] <= observation_period_end]

    transactions['period'] = transactions[datetime_col].dt.to_period(freq)

    # Mark first transactions
    transactions['first'] = transactions.groupby(customer_id_col)[datetime_col].transform('min') == transactions[datetime_col]

    # Initialize high season transaction counts to zero
    transactions['high_season_tx'] = 0
    if high_season_col:
        # Only count as high season if it's not the first transaction
        transactions.loc[~transactions['first'] & (transactions[high_season_col] == 1), 'high_season_tx'] = 1

    aggregation_functions = {
        'first': 'max',  # To identify first transactions
        'high_season_tx': 'sum',  # Sum high season transactions
    }
    if monetary_value_col:
        aggregation_functions[monetary_value_col] = 'sum'  # Aggregate monetary value if provided

    aggregated_data = transactions.groupby([customer_id_col, 'period'], as_index=False).agg(aggregation_functions)

    # Reset 'datetime_col' to reflect the aggregated period
    aggregated_data[datetime_col] = aggregated_data['period'].apply(lambda x: x.start_time)
    aggregated_data.drop(columns=['period'], inplace=True)

    return aggregated_data


def summary_data_from_transaction_data(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
    freq_multiplier=1,
    include_first_transaction=False,
):
    """
    Return summary data from transactions.

    This transforms a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a DataFrame of the form:
        customer_id, frequency, recency, T [, monetary_value]

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the columns in the transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    observation_period_end: datetime, optional
         a string or datetime to denote the final date of the study.
         Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    freq: string, optional
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    freq_multiplier: int, optional
        Default: 1. Useful for getting exact recency & T. Example:
        With freq='D' and freq_multiplier=1, we get recency=591 and T=632
        With freq='h' and freq_multiplier=24, we get recency=590.125 and T=631.375
    include_first_transaction: bool, optional
        Default: False
        By default the first transaction is not included while calculating frequency and
        monetary_value. Can be set to True to include it.
        Should be False if you are going to use this data with any fitters in lifetimes package

    Returns
    -------
    :obj: DataFrame:
        customer_id, frequency, recency, T [, monetary_value]
    """

    if observation_period_end is None:
        observation_period_end = (
            pd.to_datetime(transactions[datetime_col].max(), format=datetime_format).to_period(freq).to_timestamp()
        )
    else:
        observation_period_end = (
            pd.to_datetime(observation_period_end, format=datetime_format).to_period(freq).to_timestamp()
        )

    # label all of the repeated transactions
    repeated_transactions = _find_first_transactions(
        transactions, customer_id_col, datetime_col, monetary_value_col, datetime_format, observation_period_end, freq
    )
    # reset datetime_col to timestamp
    repeated_transactions[datetime_col] = pd.Index(repeated_transactions[datetime_col]).to_timestamp()

    # count all orders by customer.
    customers = repeated_transactions.groupby(customer_id_col, sort=False)[datetime_col].agg(["min", "max", "count"])

    if not include_first_transaction:
        # subtract 1 from count, as we ignore their first order.
        customers["frequency"] = customers["count"] - 1
    else:
        customers["frequency"] = customers["count"]

    customers["T"] = (observation_period_end - customers["min"]) / np.timedelta64(1, freq) / freq_multiplier
    customers["recency"] = (customers["max"] - customers["min"]) / np.timedelta64(1, freq) / freq_multiplier

    summary_columns = ["frequency", "recency", "T"]

    if monetary_value_col:
        if not include_first_transaction:
            # create an index of all the first purchases
            first_purchases = repeated_transactions[repeated_transactions["first"]].index
            # by setting the monetary_value cells of all the first purchases to NaN,
            # those values will be excluded from the mean value calculation
            repeated_transactions.loc[first_purchases, monetary_value_col] = np.nan
        customers["monetary_value"] = (
            repeated_transactions.groupby(customer_id_col)[monetary_value_col].mean().fillna(0)
        )
        summary_columns.append("monetary_value")

    return customers[summary_columns].astype(float)


def summary_data_from_transaction_data_season(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    high_season_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
    freq_multiplier=1,
    include_first_transaction=False,
):
    if observation_period_end is None:
        observation_period_end = pd.to_datetime(transactions[datetime_col].max(), format=datetime_format).to_period(freq).to_timestamp()
    else:
        observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format).to_period(freq).to_timestamp()

    repeated_transactions = _find_first_transactions_season(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        high_season_col,  # Pass the high season column
        datetime_format,
        observation_period_end,
        freq
    )

    # Prepare aggregation
    agg_dict = {
        datetime_col: ['min', 'max', 'count'],
    }
    if high_season_col:
        agg_dict["high_season_col"] = 'sum'
    if monetary_value_col:
        agg_dict[monetary_value_col] = 'mean'

    # Group by customer and aggregate data
    customers = repeated_transactions.groupby(customer_id_col, sort=False).agg(agg_dict)

    # Flatten the MultiIndex columns created by agg
    customers.columns = ['_'.join(col).strip('_') for col in customers.columns.values]

    # Adjust frequency for the inclusion or exclusion of the first transaction
    customers["frequency"] = customers[datetime_col + "_count"] - 1 if not include_first_transaction else customers[datetime_col + "_count"]

    # Calculate T and recency
    customers["T"] = (observation_period_end - pd.to_datetime(customers[datetime_col + "_min"])).dt.days / freq_multiplier
    customers["recency"] = (pd.to_datetime(customers[datetime_col + "_max"]) - pd.to_datetime(customers[datetime_col + "_min"])).dt.days / freq_multiplier

    # Include monetary_value if specified
    if monetary_value_col:
        customers.rename(columns={f'{monetary_value_col}_mean': 'monetary_value'}, inplace=True)

    # Reset index to ensure customer_id_col is a column
    customers.reset_index(inplace=True)

    # Fill NaNs with zeros, particularly for high_season_tx if high_season_col wasn't provided
    customers.fillna(0, inplace=True)

    # Ensure columns are in the correct data type
    type_cast_dict = {'frequency': 'int', 'T': 'float', 'recency': 'float'}
    if 'high_season_col' in customers.columns:
        type_cast_dict['high_season_col'] = 'int'
    if monetary_value_col and 'monetary_value' in customers.columns:
        type_cast_dict['monetary_value'] = 'float'
    customers = customers.astype(type_cast_dict)


    return customers


