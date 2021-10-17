
def window_combination(df, look_back):
    all_combinations = []
    for ship_id in df['Ship.to'].unique():
        relevant_indices = df.loc[df['Ship.to']==ship_id].index
        for i in range(len(relevant_indices)-look_back):
            current_group = relevant_indices[i:i+look_back].tolist()
            index_to_pred = relevant_indices[i+look_back]
            all_combinations.append((current_group, index_to_pred))
    return all_combinations
