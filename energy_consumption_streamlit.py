# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import math

# # Function to create the dataset for LSTM
# def create_dataset(data, time_step=1):
#     X, Y = [], []
#     for i in range(len(data) - time_step - 1):
#         a = data[i:(i + time_step), 0]
#         X.append(a)
#         Y.append(data[i + time_step, 0])
#     return np.array(X), np.array(Y)

# # Streamlit App Title
# st.title("Energy Consumption Prediction Using LSTM")

# # File Upload
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Uploaded Dataset:")
#     st.write(df.head())

#     # EDA Options
#     st.sidebar.header("Exploratory Data Analysis")
#     if st.sidebar.checkbox("Show dataset info"):
#         st.write(df.info())
#     if st.sidebar.checkbox("Show dataset statistics"):
#         st.write(df.describe())

#     # Selecting the column for prediction
#     target_column = st.selectbox("Select the target column for prediction", df.columns)

#     # Plotting the target variable
#     st.write("### Target Column Visualization")
#     st.line_chart(df[target_column])

#     # Preprocessing the data
#     data = df[[target_column]].values
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     # Splitting data into training and testing sets
#     train_size = int(len(data_scaled) * 0.8)
#     test_size = len(data_scaled) - train_size
#     train_data, test_data = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

#     st.write(f"Training data size: {train_size}")
#     st.write(f"Testing data size: {test_size}")

#     # Hyperparameter Selection
#     st.sidebar.header("LSTM Model Hyperparameters")
#     time_step = st.sidebar.slider("Time Step", min_value=1, max_value=50, value=10)
#     lstm_units = st.sidebar.slider("Number of LSTM Units", min_value=10, max_value=200, value=50)
#     epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=20)
#     batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=128, value=32)

#     # Creating dataset for LSTM
#     X_train, y_train = create_dataset(train_data, time_step)
#     X_test, y_test = create_dataset(test_data, time_step)

#     # Reshaping input to be [samples, time steps, features]
#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     # Building the LSTM model
#     model = Sequential()
#     model.add(LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1)))
#     model.add(LSTM(lstm_units, return_sequences=False))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     # Training the model
#     with st.spinner('Training the LSTM model...'):
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#     # Making predictions
#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)

#     # Inverse transforming the predictions
#     train_predict = scaler.inverse_transform(train_predict)
#     test_predict = scaler.inverse_transform(test_predict)
#     y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
#     y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

#     # Calculating RMSE
#     train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
#     test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
#     st.write(f"Train RMSE: {train_rmse}")
#     st.write(f"Test RMSE: {test_rmse}")

#     # Plotting the predictions
#     st.write("### Predictions vs Actual Values")
#     fig, ax = plt.subplots()
#     ax.plot(scaler.inverse_transform(data_scaled), label='Actual Data')
#     train_plot = np.empty_like(data_scaled)
#     train_plot[:, :] = np.nan
#     train_plot[time_step:len(train_predict) + time_step, :] = train_predict
#     ax.plot(train_plot, label='Train Prediction')

#     test_plot = np.empty_like(data_scaled)
#     test_plot[:, :] = np.nan
#     test_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
#     ax.plot(test_plot, label='Test Prediction')
#     plt.legend()
#     st.pyplot(fig)



# import streamlit as st 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import math

# # Function to create the dataset for LSTM
# def create_dataset(data, time_step=1):
#     X, Y = [], []
#     for i in range(len(data) - time_step - 1):
#         a = data[i:(i + time_step), 0]
#         X.append(a)
#         Y.append(data[i + time_step, 0])
#     return np.array(X), np.array(Y)

# # Streamlit App Title
# st.title("Energy Consumption Prediction Using LSTM")

# # File Upload
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Uploaded Dataset:")
#     st.write(df.head())

#     # EDA Options
#     st.sidebar.header("Exploratory Data Analysis")
#     if st.sidebar.checkbox("Show dataset info"):
#         st.write(df.info())
#     if st.sidebar.checkbox("Show dataset statistics"):
#         st.write(df.describe())

#     # Selecting the column for prediction
#     target_column = st.selectbox("Select the target column for prediction", df.columns)

#     # Plotting the target variable
#     st.write("### Target Column Visualization")
#     st.line_chart(df[target_column])

#     # Preprocessing the data
#     data = df[[target_column]].values
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     # Splitting data into training and testing sets
#     train_size = int(len(data_scaled) * 0.8)
#     test_size = len(data_scaled) - train_size
#     train_data, test_data = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

#     st.write(f"Training data size: {train_size}")
#     st.write(f"Testing data size: {test_size}")

#     # Hyperparameter Selection
#     st.sidebar.header("LSTM Model Hyperparameters")
#     time_step = st.sidebar.slider("Time Step", min_value=1, max_value=50, value=10)
#     lstm_units = st.sidebar.slider("Number of LSTM Units", min_value=10, max_value=200, value=50)
#     epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=20)
#     batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=128, value=32)

#     # Creating dataset for LSTM
#     X_train, y_train = create_dataset(train_data, time_step)
#     X_test, y_test = create_dataset(test_data, time_step)

#     # Reshaping input to be [samples, time steps, features]
#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     # Button to start training
#     if st.button("Start Training"):
#         # Building the LSTM model
#         model = Sequential()
#         model.add(LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1)))
#         model.add(LSTM(lstm_units, return_sequences=False))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')

#         # Training the model
#         with st.spinner('Training the LSTM model...'):
#             model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#         # Making predictions
#         train_predict = model.predict(X_train)
#         test_predict = model.predict(X_test)

#         # Inverse transforming the predictions
#         train_predict = scaler.inverse_transform(train_predict)
#         test_predict = scaler.inverse_transform(test_predict)
#         y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
#         y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

#         # Calculating RMSE
#         train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
#         test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
#         st.write(f"Train RMSE: {train_rmse}")
#         st.write(f"Test RMSE: {test_rmse}")

#         # Plotting the predictions
#         st.write("### Predictions vs Actual Values")
#         fig, ax = plt.subplots()
#         ax.plot(scaler.inverse_transform(data_scaled), label='Actual Data')
#         train_plot = np.empty_like(data_scaled)
#         train_plot[:, :] = np.nan
#         train_plot[time_step:len(train_predict) + time_step, :] = train_predict
#         ax.plot(train_plot, label='Train Prediction')

#         test_plot = np.empty_like(data_scaled)
#         test_plot[:, :] = np.nan
#         test_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
#         ax.plot(test_plot, label='Test Prediction')
#         plt.legend()
#         st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import math
# import io  # For capturing output of df.info()

# # Function to create the dataset for LSTM
# def create_dataset(data, time_step=1):
#     X, Y = [], []
#     for i in range(len(data) - time_step - 1):
#         a = data[i:(i + time_step), 0]
#         X.append(a)
#         Y.append(data[i + time_step, 0])
#     return np.array(X), np.array(Y)

# # Streamlit App Title
# st.title("Energy Consumption Prediction Using LSTM")

# # File Upload
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Uploaded Dataset:")
#     st.write(df.head())

#     # EDA Options
#     st.sidebar.header("Exploratory Data Analysis")
#     if st.sidebar.checkbox("Show dataset info"):
#         buffer = io.StringIO()
#         df.info(buf=buffer)  # Capture the info output in the buffer
#         info = buffer.getvalue()  # Get the string value from the buffer
#         st.text(info)  # Display the captured info in Streamlit
#     if st.sidebar.checkbox("Show dataset statistics"):
#         st.write(df.describe())

#     # Selecting the column for prediction
#     target_column = st.selectbox("Select the target column for prediction", df.columns)

#     # Plotting the target variable
#     st.write("### Target Column Visualization")
#     st.line_chart(df[target_column])

#     # Preprocessing the data
#     data = df[[target_column]].values
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     # Splitting data into training and testing sets
#     train_size = int(len(data_scaled) * 0.8)
#     test_size = len(data_scaled) - train_size
#     train_data, test_data = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

#     st.write(f"Training data size: {train_size}")
#     st.write(f"Testing data size: {test_size}")

#     # Hyperparameter Selection
#     st.sidebar.header("LSTM Model Hyperparameters")
#     time_step = st.sidebar.slider("Time Step", min_value=1, max_value=50, value=10)
#     lstm_units = st.sidebar.slider("Number of LSTM Units", min_value=10, max_value=200, value=50)
#     epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=20)
#     batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=128, value=32)

#     # Creating dataset for LSTM
#     X_train, y_train = create_dataset(train_data, time_step)
#     X_test, y_test = create_dataset(test_data, time_step)

#     # Reshaping input to be [samples, time steps, features]
#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     # Button to start training
#     if st.button("Start Training"):
#         # Building the LSTM model
#         model = Sequential()
#         model.add(LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1)))
#         model.add(LSTM(lstm_units, return_sequences=False))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')

#         # Training the model
#         with st.spinner('Training the LSTM model...'):
#             model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#         # Making predictions
#         train_predict = model.predict(X_train)
#         test_predict = model.predict(X_test)

#         # Inverse transforming the predictions
#         train_predict = scaler.inverse_transform(train_predict)
#         test_predict = scaler.inverse_transform(test_predict)
#         y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
#         y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

#         # Calculating RMSE
#         train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
#         test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
#         st.write(f"Train RMSE: {train_rmse}")
#         st.write(f"Test RMSE: {test_rmse}")

#         # Plotting the predictions
#         st.write("### Predictions vs Actual Values")
#         fig, ax = plt.subplots()
#         ax.plot(scaler.inverse_transform(data_scaled), label='Actual Data')
#         train_plot = np.empty_like(data_scaled)
#         train_plot[:, :] = np.nan
#         train_plot[time_step:len(train_predict) + time_step, :] = train_predict
#         ax.plot(train_plot, label='Train Prediction')

#         test_plot = np.empty_like(data_scaled)
#         test_plot[:, :] = np.nan
#         test_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
#         ax.plot(test_plot, label='Test Prediction')
#         plt.legend()
#         st.pyplot(fig)




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
import io  # For capturing output of df.info()

# Function to create the dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to parse df.info() output
def parse_dataframe_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)  # Capture the info output in the buffer
    info_lines = buffer.getvalue().splitlines()  # Split the output into lines
    column_data = []

    for line in info_lines[5:-2]:  # Relevant lines start at index 5, end 2 lines from the bottom
        parts = line.split()
        column_name = " ".join(parts[1:-2])  # Handle multi-word column names
        non_null_count = parts[-2]
        dtype = parts[-1]
        column_data.append({"Column Name": column_name, "Non-Null Count": non_null_count, "Data Type": dtype})

    memory_usage = info_lines[-1]  # Last line contains memory usage
    return pd.DataFrame(column_data), memory_usage

# Streamlit App Title
st.title("Energy Consumption Prediction Using LSTM")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset:")
    st.write(df.head())

    # EDA Options
    st.sidebar.header("Exploratory Data Analysis")
    if st.sidebar.checkbox("Show dataset info"):
        parsed_info, memory_usage = parse_dataframe_info(df)  # Parse the dataframe info
        st.write("### Dataset Information:")
        st.dataframe(parsed_info)  # Display as a dataframe
        st.write(f"**Memory Usage:** {memory_usage}")
    if st.sidebar.checkbox("Show dataset statistics"):
        st.write(df.describe())

    # Selecting the column for prediction
    target_column = st.selectbox("Select the target column for prediction", df.columns)

    # Plotting the target variable
    st.write("### Target Column Visualization")
    st.line_chart(df[target_column])

    # Preprocessing the data
    data = df[[target_column]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Splitting data into training and testing sets
    train_size = int(len(data_scaled) * 0.8)
    test_size = len(data_scaled) - train_size
    train_data, test_data = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

    st.write(f"Training data size: {train_size}")
    st.write(f"Testing data size: {test_size}")

    # Hyperparameter Selection
    st.sidebar.header("LSTM Model Hyperparameters")
    time_step = st.sidebar.slider("Time Step", min_value=1, max_value=50, value=10)
    lstm_units = st.sidebar.slider("Number of LSTM Units", min_value=10, max_value=200, value=50)
    epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=20)
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=128, value=32)

    # Creating dataset for LSTM
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshaping input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Button to start training
    if st.button("Start Training"):
        with st.spinner('Training the LSTM model...'):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Making predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transforming the predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculating RMSE
        train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
        st.write(f"Train RMSE: {train_rmse}")
        st.write(f"Test RMSE: {test_rmse}")

        # Plotting the predictions
        st.write("### Predictions vs Actual Values")
        fig, ax = plt.subplots()
        ax.plot(scaler.inverse_transform(data_scaled), label='Actual Data')
        train_plot = np.empty_like(data_scaled)
        train_plot[:, :] = np.nan
        train_plot[time_step:len(train_predict) + time_step, :] = train_predict
        ax.plot(train_plot, label='Train Prediction')

        test_plot = np.empty_like(data_scaled)
        test_plot[:, :] = np.nan
        test_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
        ax.plot(test_plot, label='Test Prediction')
        plt.legend()
        st.pyplot(fig)
