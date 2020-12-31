if __name__ == '__main__':

    from tsfresh.examples import load_robot_execution_failures
    df, _ = load_robot_execution_failures()
    print(df)
