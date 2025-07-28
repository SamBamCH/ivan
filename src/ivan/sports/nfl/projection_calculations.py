import pandas as pd


def _devig_odds(df):
    """
    Adjusts odds data to remove vigorish and normalize probabilities.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        pd.DataFrame: DataFrame with normalized probabilities.
    """
    grouped = df.groupby(['market', 'description', 'point', 'sportsbook'])

    def devig_group(group):
        group['implied_prob'] = group['odds'].apply(lambda x: 100 / (100 + x) if x > 0 else -x / (-x + 100))
        total_prob = group['implied_prob'].sum()

        if total_prob < 1.0:
            missing_prob = 1.0 - total_prob
            adjusted_vig = 0.05
            total_prob += adjusted_vig + missing_prob

        group['normalized_prob'] = group['implied_prob'] / total_prob
        return group

    df = pd.concat([devig_group(group) for _, group in grouped])
    return df

def _calculate_weighted_probabilities(df, market_weights):
    """
    Calculates weighted probabilities based on market weights.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        pd.DataFrame: DataFrame with weighted probabilities.
    """
    grouped = df.groupby(['market', 'outcome', 'description'])

    def weighted_prob_group(group):
        market = group['market'].iloc[0]
        weights_and_rows = [(market_weights[market].get(row['sportsbook'], 0), row) for _, row in group.iterrows()]
        weights_and_rows = [(w, row) for w, row in weights_and_rows if w > 0]

        if len(weights_and_rows) < 3:
            return pd.DataFrame()

        available_weights, filtered_rows = zip(*weights_and_rows)
        total_weight = sum(available_weights) or 1

        normalized_weights = [w / total_weight for w in available_weights]
        weighted_prob = sum(
            filtered_rows[i]['normalized_prob'] * normalized_weights[i] for i in range(len(filtered_rows)))

        filtered_group = pd.DataFrame(filtered_rows)
        filtered_group['weighted_prob'] = weighted_prob
        return filtered_group

    df = pd.concat([weighted_prob_group(group) for _, group in grouped]).reset_index(drop=True)
    return df

def _calculate_consensus_value(self, df, bucket_sizes):
    """
    Calculates consensus values by bucketing and averaging.

    Args:
        df (pd.DataFrame): DataFrame with weighted probabilities.
        bucket_sizes (dict): Dictionary with bucket sizes for each market.

    Returns:
        pd.DataFrame: DataFrame with consensus values.
    """
    grouped = df.groupby(['market', 'outcome', 'description'])

    def consensus_group(group):
        market = group['market'].iloc[0]
        bucket_size = bucket_sizes.get(market, 1)

        group['point'] = pd.to_numeric(group['point'], errors='coerce')

        if group['point'].isnull().all():
            print(f"All points are NaN for market: {market}")
            return pd.DataFrame()

        group['bucket'] = (group['point'] // bucket_size) * bucket_size

        bucketed = group.groupby('bucket')

        if bucketed.ngroups == 0:
            print(f"No buckets for market: {market}")
            return pd.DataFrame()

        def bucket_aggregate(bucket_group):
            if len(bucket_group['sportsbook'].unique()) < 3:
                return pd.DataFrame()

            avg_point = bucket_group['point'].mean() if market != 'player_anytime_td' else bucket_group['point'].iloc[0]
            avg_prob = bucket_group['normalized_prob'].mean()
            bucket_group['consensus_point'] = avg_point
            bucket_group['consensus_prob'] = avg_prob
            return bucket_group

        aggregated = [bucket_aggregate(bucket_group) for _, bucket_group in bucketed]
        return pd.concat(aggregated) if aggregated else pd.DataFrame()

    df = pd.concat([consensus_group(group) for _, group in grouped]).reset_index(drop=True)
    return df

def _parse_odds_data(odds_data_json):
        """
        Parses JSON odds data into a DataFrame.

        Args:
            odds_data_json (dict): JSON data containing odds information.

        Returns:
            pd.DataFrame: DataFrame with parsed odds data.
        """
        all_rows = []
        for event_id, event_data in odds_data_json.items():
            for bookmaker in event_data['bookmakers']:
                sportsbook = bookmaker['title']
                for market in bookmaker['markets']:
                    market_key = market['key']
                    for outcome in market['outcomes']:
                        point = outcome.get('point', None)
                        if market_key == 'player_anytime_td':
                            if point is None:
                                point = 0.5
                            outcome_name = 'Over' if outcome['name'] == 'Yes' else outcome['name']
                        else:
                            outcome_name = outcome['name']

                        row = {
                            'sportsbook': sportsbook,
                            'market': market_key,
                            'description': outcome.get('description'),
                            'point': point,
                            'odds': outcome['price'],
                            'outcome': outcome_name
                        }
                        all_rows.append(row)

        return pd.DataFrame(all_rows)
    
def calculate_concensus_values(odds_data_json):
    """
    Calculates consensus values based on provided odds data.

    Args:
        odds_data_json (dict): JSON data containing odds information.

    Returns:
        pd.DataFrame: Processed DataFrame with consensus values.
    """
    df = _parse_odds_data(odds_data_json)
    bucket_sizes = {
        'player_pass_yds': 15,
        'player_rush_yds': 3,
        'player_reception_yds': 3
    }

    df = _devig_odds(df)
    df = _calculate_weighted_probabilities(df)
    df = _calculate_consensus_value(df, bucket_sizes)

    return df