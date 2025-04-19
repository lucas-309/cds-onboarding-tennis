import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.utils import resample



# TODO


def main():
    st.title("CDS Onboarding Website")
    st.header("Team Name: TBD")
    st.header("Team Members: Lucas He, Rohit Vakkalagadda, Kaitlyn Lu")
    st.divider()
    st.header("About Us")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Lucas")
        st.image("images/lucas.png", use_container_width=True)
        st.text(
            "I'm a freshman studying CS and China Asia-Pacific Studies.\n"
            "I love exploring campus\n"
            "My favorite place to eat in Collegetown: Oishii Bowl\n"
            "Contact: th689@cornell.edu"
        )

    with col2:
        st.subheader("Rohit")
        st.image("images/rohit.JPG", use_container_width=True)  # Replace with correct image path
        st.text("I'm a freshman studying CS.\n"
            "I'm on the DE subteam\n"
            "My favorite place to eat in Collegetown: Xi'An Street Food\n"
            "Contact: rv324@cornell.edu"
        )

    with col3:
        st.subheader("Kaitlyn")
        st.image("images/kaitlyn.png", use_container_width=True)  # Replace with correct image path
        st.text(
            "I'm a freshman in A&S studying CS.\n"
            "I love reading and writing ✍️📚\n"
            "My favorite place to eat in Collegetown: Gangnam Station\n"
            "Contact: kl995@cornell.edu"
        )


def introduction():
    st.title("Project Introduction")
    st.text("Our question: Given the game-specific statistics of a match, can we determine the surface on which it was played?")
    st.text("We chose this question because we all like tennis or have played tennis in high school, and thought that entering a field with tons of numerical data (like sports) would lend us better results than our previous field (crime)\n"
    )
    st.text(
        "Our dataset describes all the matches from the 2024 ATP Tour (including information like match winner, game points, etc.). "
        "We chose it as we were interested in predicting the surface that a match was played on, using the match statistics, and we thought the data from 2024 would be the most useful. "
        "Important features include the ranking points of the winner and loser at the time of the match (winner_rank_points and loser_rank_points, respectively), "
        "the ages of both players (winner_age, loser_age), and point information (such as points played by the winner and loser)."
    )
    st.text("We want to use this data to make a prediction model (based on power rankings) in order to try and predict the surface a match is played on given any arbitrary statistics")
    st.text(
            "Here are the features we had: \n"
            "tourney_id: Unique tournament event identifier code.\n"
            "tourney_name: Official tournament event title, e.g. Wimbledon Championships.\n"
            "surface: Court surface type, e.g. grass, clay, hard.\n"
            "draw_size: Number of players or teams in main draw.\n"
            "tourney_level: Tournament category or tier, e.g. Grand Slam.\n"
            "tourney_date: Tournament start date in YYYYMMDD format.\n"
            "match_num: Match sequence number within the tournament.\n"
            "winner_id: Unique identifier for winning player.\n"
            "winner_name: Full name of the match winner.\n"
            "winner_hand: Winner’s playing hand: Left, Right, or Unknown.\n"
            "winner_ht: Winner’s height measurement in centimeters.\n"
            "winner_ioc: Winner’s nationality IOC country code.\n"
            "winner_age: Winner’s age at match start date.\n"
            "loser_id: Unique identifier for losing match player.\n"
            "loser_name: Full name of the match loser.\n"
            "loser_hand: Loser’s playing hand: Left, Right, or Unknown.\n"
            "loser_ht: Loser’s height measurement in centimeters.\n"
            "loser_ioc: Loser’s nationality IOC country code.\n"
            "loser_age: Loser’s age at match start date.\n"
            "score: Match score by sets as formatted string.\n"
            "best_of: Total sets required to win the match.\n"
            "round: Tournament round code, e.g. QF, SF, Final.\n"
            "minutes: Total match duration measured in minutes.\n"
            "w_ace: Number of aces served by the winner.\n"
            "w_df: Number of double faults committed by winner.\n"
            "w_svpt: Winner’s total number of service points played.\n"
            "w_1stIn: Winner’s count of successful first serves in.\n"
            "w_1stWon: Number of first‑serve points won by winner.\n"
            "w_2ndWon: Number of second‑serve points won by winner.\n"
            "w_SvGms: Winner’s count of service games held.\n"
            "w_bpSaved: Number of break points saved by winner.\n"
            "w_bpFaced: Number of break points faced by winner.\n"
            "l_ace: Number of aces served by the loser.\n"
            "l_df: Number of double faults committed by loser.\n"
            "l_svpt: Loser’s total number of service points played.\n"
            "l_1stIn: Loser’s count of successful first serves in.\n"
            "l_1stWon: Number of first‑serve points won by loser.\n"
            "l_2ndWon: Number of second‑serve points won by loser.\n"
            "l_SvGms: Loser’s count of service games held.\n"
            "l_bpSaved: Number of break points saved by loser.\n"
            "l_bpFaced: Number of break points faced by loser.\n"
            "winner_rank: Winner’s world ranking at tournament start.\n"
            "winner_rank_points: Winner’s total ATP/WTA ranking points at start.\n"
            "loser_rank: Loser’s world ranking at tournament start.\n"
            "loser_rank_points: Loser’s total ATP/WTA ranking points at start.\n"
    )

    


def manipulation():
    st.title("Data Manipulation & Cleaning")
    st.text(
        "We decided to clean the data by removing all rows with NaN values. "
        "We made this decision because we knew that those NaN values would do nothing but harm our prediction model, "
        "so we decided the best choice was to remove the match entirely."
    )
    st.text(
        "In addition, we removed certain columns which had no influence on the court surface, such as tourney_date"
    )
    st.image("images/before_nan.png")
    st.text(
        "Before removing NaN values"   
    )
    st.image("images/after_nan.png")
    st.text("After remove NaN values")
    st.text("After cleaning null values and rows, we were left with 2761 rows, or 89% of the original dataset. "
        "We also label encoded the following columns: "
        "['tourney_name', 'winner_name', 'loser_name', 'surface', 'winner_hand', 'loser_hand', "
        "'tourney_level', 'winner_ioc', 'loser_ioc', 'round'].")
    st.header("Undersampling")
    st.text(
        "Later on, in classification, we noticed there are much more data points with hard courts versus clay or grass courts. "
        "We fixed this by manually undersampling hard count by filtering the data with hard courts and using scikit's resampling tool "
        "to place higher emphasis on minority data."    
    )
    st.image("images/before_undersampling.png")
    st.text(
        "Before undersampling"    
    )
    st.image("images/after_undersampling.png")
    st.text("After undersampling")



def visualization():
    st.title("Data Visualization")
    st.image("../plots/histogram_w_svpt.png")
    st.text(
        "This is a histogram of the frequency of the total service points played by the winner in each match. "
        "As shown by the graph, most winners played between 50 and 75 service points in a match."
    )
    st.image("../plots/histogram_w_ace.png")
    st.text(
        "This is a histogram of the frequency of the number of aces served by the winner in each match. "
        "As shown by the graph, most winners played under 10 aces. The graph is surprisingly bimodal."
    )
    st.text("The visualizations helped us see the distribution of winners' statistics, such as how many service points they played and how many aces they served.")
    st.subheader("Insights")
    st.text("Some interesting results we found:\n"
            "1. Based on winner age and violin plots, we found that one tournament (the Monte Carlo Masters) had older winners on average - interesting observation\n"
            "2. There was a clear spike for younger players and having more rank points on tour.\n"
            "3. Break points saved was a key metric in determining a winner (0.6 correlation) of winner or loser\n"
            "4. The average age of a winner on tour was around 26- not that important because it actually reflected the average age of a player on tour"
            )

def supervisedLearning():
    st.title("Supervised Learning")
    st.subheader("PCA")
    st.image("../ms5plots/output_pca_analysis.png")
    st.text("PCA stands for principle component analysis. We take all features of the supervised learning algorithm, and then determine the dimension on which this data has the highest variance. Then, we project all these points onto this dimensions (consider this to be a 2-D graph onto a 1-D line) and we are left with a line representation of the data. Next, we take the dimension with the next highest variance that is orthogonal to the first component and we also project our data on that line to reduce our dimensionality.")
    st.text(
    "This visualization shows the results of our PCA analysis. "
    "It helps illustrate how much variance is captured by each principal component, "
    "and how effectively the data is separated in the lower-dimensional feature space."
    )
    st.text("After creating too many principal components, the component doesn’t capture the variance but rather noise that harms accuracy")

    st.subheader("KNN")
    st.text("A K-nearest neighbors algorithm plots all data points by using the features, and then clusters them into groups with k neighbors. It is a classification algorithm.")
    st.image("../ms5plots/output_knn_kneighbors.png")
    st.text(
    "Here we examine how different values of K (number of neighbors) affect KNN performance. "
    "As shown, the choice of K can drastically change the boundary of classification or regression fits."
    )

    # 3) Different Scaling Factors
    st.image("../ms5plots/different_scaling_factors.png")
    st.text(
    "This plot demonstrates the effect of various scaling techniques (e.g., StandardScaler, MinMaxScaler). "
    "We can see how different scalers influence feature distributions and, consequently, model performance."
    )

    st.subheader("SVM")
    st.text("A support vector machine (SVM) is a supervised machine learning algorithm that finds the best boundary (called a hyperplane) to separate different classes of data points with the maximum margin (the largest possible distance from the nearest data points, called support vectors). They are used for classification and regression. SVMs use kernel functions, which map input data into a higher-dimensional space so that a linear separator can be found (examples include linear kernels and polynomial kernels.")
    st.image("../ms5plots/changing_gamma_and_c.png")
    st.text(
        "This graphic illustrates how adjusting the gamma and C hyperparameters in an SVM affects the decision boundary. "
        "Higher gamma values can overfit by creating very wiggly boundaries, whereas different C values regulate margin size."
    )
    st.text("Increasing C means that the SVM will try to minimize misclassifications of training points (leading to higher accuracy) and will lead to a smaller margin size. As seen in our “Test Accuracy vs. C” graph, as we increase the value of C from 10-2 to 102, the accuracy goes from under 0.66 to nearly 0.70.")
    st.text("Accuracy increases as C does because the SVM will penalize misclassifications (in training data) heavily, so the model will fit the training data more accurately. However, when C gets astronomically large, it could end up overfitting to the training data and contorting the decision boundary to account for random noise. This means that the model’s ability to generalize to new data will plummet and thus will perform poorly on the test data.")

    # 5) Changing C Linearly
    st.image("../ms5plots/changing_c_linearly.png")
    st.text(
        "This plot focuses on systematically increasing the C parameter in an SVM. "
        "We can see how a larger C penalizes misclassifications more aggressively, potentially reducing bias but increasing variance."
    )
    st.text("One of the most obvious signs of the SVM overfitting is when the training accuracy is high but the test accuracy is low. Another sign is when the decision boundary is highly complex, which could indicate that random noise and outliers in the data are affecting the boundary.")

    st.subheader("Decision Tree")
    st.text("A decision tree is a supervised machine learning algorithm that resembles a flowchart. Each internal node is a question about a feature of the data, and the branches represent the possible outcomes. Leaf nodes are the final predictions given by the algorithm, whether that would be a label (for classification) or a value (for regression).")
    st.text("As max depth increases, the model will make increasingly complex decisions in order to match the training data as much as possible. This comes at the cost of test data accuracy, because the model will be overfitted to the training data and thus unable to accurately predict with new data.")
    st.text("Our random forest (RF) model is an enhanced decision tree that averages many decision trees for the best results")

    st.subheader("Best model")
    st.image("images/models.png")
    st.text("The test errors were as follows: \n"
            "K-NN - see graph, but around 0.63\n"
            "SVM - see correlation graph, but best was Gamma = 0.01, C = 10.0, accuracy of 0.69\n"
            "RF - highest test accuracy with 0.7034\n"
            "Training time: \n"
            "Fastest: KNN\n"
            "SVM and RFC were comparable \n"
            "Inference time: \n"
            "Fastest: RFC \n"
            "SVM and KNN were comparable"
            )


def conclusion():
    st.title("Conclusion")
    st.text(
        "In our project, we managed to build several machine learning models to accurately predict features of a game given its other characteristics. "
        "In particular, we tried to predict the type of surface the game was played on, given the player data and the game score."
        "We achieved a 70% accuracy with our Random Forest model."
    )
    st.text(
        "We gained hands-on experience with real-world data preprocessing, data cleaning, and the implementation of various supervised learning models. "
        "We also learned how to interpret the influence of model hyperparameters such as K in KNN and gamma and C in SVMs. " 
        "Visualization was essential for understanding the internal mechanics of the models and conveying our findings clearly."
    )
    st.text("If we were to do the project again, we would:\n"
            "- Start with clearer goals for model evaluation \n"
            "- Be more careful with selecting a quality dataset \n"
            "- Be more proactive in recognizing when we may need to pivot to a different dataset"
            )

def frontend():
    st.title("Frontend")
    st.text("We are using Streamlit for the frontend in order to display our project's progress and highlights.")
    st.image("pages.png")
    st.text("A glimpse of the code used for the frontend.")


def label_encode(column):
    labels = column.unique()
    label_dict = {label: idx for idx, label in enumerate(labels)}
    return column.map(label_dict), label_dict

def testRFModel():
    st.title("Test Random Forest Model")


    st.markdown("Enter match statistics to predict the **surface** type (hard, clay, grass)")

    # Load or reuse the original dataset
    df = pd.read_csv("../atp_matches_2024.csv")

    # Undersample hard data
    df_majority = df[df['surface'] == 'Hard']
    df_minority = df[df['surface'] != 'Hard']

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    df = pd.concat([df_majority_downsampled, df_minority])


    df['surface_encoded'], surface_encoder = label_encode(df['surface'])
    inverse_surface_encoder = {v: k for k, v in surface_encoder.items()}

    # Prepare features and target
    drop_cols = ['match_num', 'tourney_date', 'round', 'tourney_id', 'score', 
                 'surface_encoded', "tourney_name", "winner_name", "loser_name", "surface", 
                 "winner_hand", "loser_hand", "tourney_level", "winner_ioc", "loser_ioc"]
    # drop_cols += ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon',
    #    'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df',
    #    'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved',
    #    'l_bpFaced']
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]) 
    y = df['surface_encoded']

    # Train/test split and model setup (use cache in production)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Dynamic form for user input
    user_input = {}
    clay_df = df[df['surface'] == 'Clay']
    st.subheader("Input Match Stats:")
    for col in X.columns:
        if col in ['draw_size', 'minutes', 'winner_id', 'loser_id', 'winner_rank_points', 'loser_rank_points']:
            val = st.number_input(f"{col}", value=float(df[col].mean()), format="%.4f")
            user_input[col] = val
        else:
            user_input[col] = clay_df[col].mean()
    
    st.write("Label Distribution in y_train:")
    st.write(y_train.value_counts())

    if st.button("Predict Surface Type"):
        input_df = pd.DataFrame([user_input], columns=X.columns)
        input_scaled = scaler.transform(input_df)
        prediction = rf.predict(input_scaled)[0]
        proba = rf.predict_proba(input_scaled)[0][prediction]

        surface_name = inverse_surface_encoder.get(prediction, "Unknown")

        # Calculate test error
        y_pred_test = rf.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_error = round(1 - test_accuracy, 4)

        st.write("Input Vector (scaled):", input_scaled)
        st.write("Raw Prediction:", prediction)

        st.success(f"🏟️ Predicted Surface: **{surface_name}**")
        st.info(f"📉 Test Set Error: **{test_error}**")
        st.info(f"🔍 Confidence Score: **{round(proba * 100, 2)}%**")
    



def ci():
    st.title("Continuous Integration")
    st.text(
        "We are using GitHub actions to run the test file every time we push code into the repository "
        "so that we can see how our code affects any preconditions/requirements we've set."
    )
    st.image("github.png")
    st.text("A screenshot of our workflow runs.")






if __name__ == "__main__":
    pg = st.navigation([
        st.Page(main, title="Homepage"),
        st.Page(introduction, title="Project Introduction"),
        st.Page(manipulation, title="Data Manipulation"),
        st.Page(visualization, title="Data Visualization"),
        st.Page(supervisedLearning, title="Supervised Learning"),
        st.Page(testRFModel, title="Test RF Model"),
        st.Page(conclusion, title="Conclusion"),
        st.Page(frontend, title="Frontend"),
        st.Page(ci, title="Continuous Integration")
    ])
    pg.run()
