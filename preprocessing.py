from sklearn.preprocessing import StandardScaler # type: ignore

def preprocess_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler
