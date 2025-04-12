import streamlit as st

# TODO


def main():
    st.title("CDS Onboarding Website")
    st.header("Team Name: TBD")
    st.header("Team Members: Lucas He, Rohit Vakkalagadda, Kaitlyn Lu")
    st.divider()
    st.header("About Us")
    st.subheader("Lucas:")
    st.text(
        "I'm a freshman studying CS and math. I love exploring new places on campus. "
        "You can find me playing basketball at Helen Newman or playing Valorant at the gaming lounge. "
        "Feel free to reach out![text + photo here]"
    )
    st.subheader("Rohit:")
    st.text("[text + photo here]")
    st.subheader("Kaitlyn:")
    st.text("I am a freshman in A&S studying CS. I love reading and writing, and I'm excited to be a part of CDS! [add photo]")


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


def frontend():
    st.title("Frontend")
    st.text("We are using Streamlit for the frontend in order to display our project's progress and highlights.")
    st.image("pages.png")
    st.text("A glimpse of the code used for the frontend.")


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
        st.Page(frontend, title="Frontend"),
        st.Page(ci, title="Continuous Integration")
    ])
    pg.run()
