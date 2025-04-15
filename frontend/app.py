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
            "I love exploring new places on campus. "
            "My favorite place to eat in Collegetown: Oishii Bowl 🍜\n"
            "Contact: th689@cornell.edu"
        )

    with col2:
        st.subheader("Rohit")
        st.image("images/rohit.png", use_container_width=True)  # Replace with correct image path
        st.text(
        )

    with col3:
        st.subheader("Kaitlyn")
        st.image("images/kaitlyn.png", use_container_width=True)  # Replace with correct image path
        st.text(
            "I am a freshman in A&S studying CS.\n"
            "I love reading and writing ✍️📚\n"
        )


def introduction():
    st.title("Project Introduction")
    st.text("Our project aims to accurately predict the winner of the 2025 Australian Open using the 2024 player data to create power rankings.")
    st.text(
        "Our dataset describes all the matches from the 2024 ATP Tour (including information like match winner, game points, etc.). "
        "We chose it as we were interested in predicting the winner of the 2025 Australian Open, and we thought the data from 2024 would be the most useful. "
        "Important features include the ranking points of the winner and loser at the time of the match (winner_rank_points and loser_rank_points, respectively), "
        "the ages of both players (winner_age, loser_age), and point information (such as points played by the winner and loser)."
    )
    st.text("We want to use this data to make a prediction model (based on power rankings) in order to try and predict the winner of the 2025 Australian Open.")


def manipulation():
    st.title("Data Manipulation")
    st.text(
        "We decided to clean the data by removing all rows with NaN values. "
        "We made this decision because we knew that those NaN values would do nothing but harm our prediction model, "
        "so we decided the best choice was to remove the match entirely."
    )


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

def supervisedLearning():
    st.title("Supervised Learning")
    st.image("../ms5plots/output_pca_analysis.png")
    st.text(
    "This visualization shows the results of our PCA analysis. "
    "It helps illustrate how much variance is captured by each principal component, "
    "and how effectively the data is separated in the lower-dimensional feature space."
    )

    # 2) KNN K-Neighbors
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

    # 4) Changing Gamma and C
    st.image("../ms5plots/changing_gamma_and_c.png")
    st.text(
        "This graphic illustrates how adjusting the gamma and C hyperparameters in an SVM affects the decision boundary. "
        "Higher gamma values can overfit by creating very wiggly boundaries, whereas different C values regulate margin size."
    )

    # 5) Changing C Linearly
    st.image("../ms5plots/changing_c_linearly.png")
    st.text(
        "This plot focuses on systematically increasing the C parameter in an SVM. "
        "We can see how a larger C penalizes misclassifications more aggressively, potentially reducing bias but increasing variance."
    )

def conclusion():
    st.title("Conclusion")
    st.text(
        "In our project, we managed to build several machine learning models to accurately predict features of a game given its other characteristics. "
        "For instance, we tried to predict the type of surface the game was played on, given the player data and the game score."
        "We achieved "
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
