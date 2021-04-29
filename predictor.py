import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

def predictRuns(input_test):
    
    # read data from csv file into a pandas dataFrame
    
    with open('all_matches.csv') as f:
        
        ipl_data = pd.read_csv(f)
    # prints all columns
    # print(data.columns)
    # all columns
    # [ 'match_id','venue','innings','ball',
    # 'batting_team', 'bowling_team','striker','non_striker','bowler',
    # 'runs_off_bat', 'extras', 'wides' ,'noballs','byes','legbyes',
    # 'penalty','wicket_type','player_dismissed','other_wicket_type',
    # 'other_player_dismissed']
    
    relevantColumns = [ 'match_id','venue','innings','ball',
                   'batting_team', 'bowling_team','striker','non_striker','bowler',
                   'runs_off_bat', 'extras', 'wides' ,'noballs','byes','legbyes',
                   'penalty','wicket_type','player_dismissed','other_wicket_type',
                   'other_player_dismissed']
    ipl_data = ipl_data[relevantColumns]
    
    # create another column that tells the number of runs scored, including off the bat and 
    # extra runs conceded by the bowling team
    
    ipl_data['total_runs'] = ipl_data['runs_off_bat']+ipl_data['extras']
    
    # now drop the columns 'run_off_bat' and 'extras' as they are not required anymore
    
    ipl_data=ipl_data.drop(columns=['runs_off_bat','extras'])
    
    # only selectrows belonging to first 6 overs
    
    ipl_data = ipl_data[ipl_data['ball']<=5.6]
    
    ipl_data = ipl_data[ipl_data['innings']<=2]
    
    # preprocess the data so that we get a tuple of following kind in each row:
    # ('match_id','venue','innings','batting_team','bowling_team','total_runs')
    
    ipl_data = ipl_data.groupby(['match_id',
                                 'venue',
                                 'innings',
                                 'batting_team',
                                 'bowling_team']).total_runs.sum()
    
    # covert back to the dataframe
    
    ipl_data = ipl_data.reset_index()
    
    ipl_data = ipl_data.drop(columns=['match_id'])
    
    ipl_data.to_csv('myPreprocessed.csv', index = False)
    
    
    # read data from the unzipped folder
    
    data = pd.read_csv('myPreprocessed.csv')
    venue_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    data['venue'] = venue_encoder.fit_transform(data['venue'])
    data['batting_team'] = team_encoder.fit_transform(data['batting_team'])
    data['bowling_team'] = team_encoder.fit_transform(data['bowling_team'])
    
    # get data in a numpy array
    
    anArray = data.to_numpy()
    
    # get independent and target variables
    
    X,y = anArray[:,:3],anArray[:,3]
    
    X = np.concatenate((np.eye(42)[anArray[:,0]],
                        np.eye(2)[anArray[:,1] -1],
                        np.eye(15)[anArray[:,2]],
                        np.eye(15)[anArray[:,3]],
                        ), axis = 1)
    
    # split data in training and testing
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.25)
    
    # fit a linear regressor
    
    linearRegressor = LinearRegression()
    
    # train the model
    
    linearRegressor.fit(X_train, y_train)
    
    # save the model and the supporting label encoders
    
    joblib.dump(linearRegressor,'regression_model.joblib')
    joblib.dump(venue_encoder,'venue_encoder.joblib')
    joblib.dump(team_encoder,'team_encoder.joblib')
    print(linearRegressor.score(X_test,y_test))

                                                    
   
    with open('regression_model.joblib','rb') as f:
        regressor = joblib.load(f)
    with open('venue_encoder.joblib','rb') as f:
        venue_encoder = joblib.load(f)
    with open('team_encoder.joblib','rb') as f:
        team_encoder = joblib.load(f)
    
    # read test data
    test_case = pd.read_csv(input_test)
    # encode venue and batting and bowling teams
    test_case['venue'] = venue_encoder.transform(test_case['venue'])
    test_case['batting_team'] = team_encoder.transform(test_case['batting_team'])
    test_case['bowling_team'] = team_encoder.transform(test_case['bowling_team'])
    
    # make sure that the order of columns is same as that fed to the model
    test_case = test_case[['venue','innings','batting_team','bowling_team']]
    
    # convert input test case into numpy array
    testArray = test_case.to_numpy()
    
    # one hot encode venue, batting and bowling teams
    test_case = np.concatenate((np.eye(42)[testArray[:,0]],
                                np.eye(2)[testArray[:,1] -1],
                                np.eye(15)[testArray[:,2]],
                                np.eye(15)[testArray[:,3]],
                                ),
                               axis = 1)
    return regressor.predict(test_case)
    
    
    
