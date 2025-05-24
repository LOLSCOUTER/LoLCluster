import pandas as pd

df = pd.read_csv("LOLCLUSTER/champion_vectors/champion_with_roles.csv")

def make_team_vector(team_df):
    team_df = team_df.sort_values(by='champion')
    vector = []
    for _, row in team_df.iterrows():
        vector += [
            row['champion'],
            row['kills'], row['deaths'], row['assists'],
            row['damage'], row['taken'], row['heal'],
            row['role_cluster']
        ]
        vector += row.filter(like='item_').tolist()
    return vector

result = []

for (match_id, team_id), group in df.groupby(['match_id', 'team_id']):
    if len(group) != 5:
        continue
    vector = make_team_vector(group)
    vector = pd.Series(vector) 
    vector['match_id'] = match_id
    vector['team_id'] = team_id
    vector['win'] = group['win'].iloc[0]
    result.append(vector)

team_vectors = pd.DataFrame(result)
team_vectors.to_csv("LOLCLUSTER/champion_vectors/team_vectors.csv", index=False)
