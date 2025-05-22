def print_introduction(data):
    print("\nWelcome! In this project, we will be working with the Student Performance dataset obtained from the UCI "
          "Machine Learning Repository.")
    print("This dataset contains information on various personal, academic, and social factors that may influence a "
          "student's academic outcomes.")
    print("For the purposes of our analysis, we will focus on a subset of 500 students to ensure fast experimentation "
          "and testing.")
    print("The goal is to predict whether a student passes or fails based on selected features.")
    print("\nThe selected features used for training the model are:")
    print("- 'studytime': Weekly study time")
    print("- 'failures': Number of past class failures")
    print("- 'schoolsup': Extra educational support (1 = yes, 0 = no)")
    print("- 'health': Current health status (1 = very bad to 5 = very good)")
    print("- 'absences': Number of school absences")
    print("\nHere is the cleaned subset of the dataset, showing only the most relevant features for our analysis:")
    print(data)
    print("\nDISCLAIMER: This code may not provide the same result every time, as some parameters, such as maximum "
          "iterations, might be insufficient and cause the model to be trained differently in each execution.\n")