# coding=utf-8
"""work as data source to model

mix the different data to a uniform interface
"""
from Src.utils import config_util
import  pandas as pd

def __get_user_profile(end_time):
    """get user profile from statistics and basic user profile
    """
    print "the basic user statistics "
    user = pd.read_csv(config_util.get(section='Path',key='User'))
    user_action_day = pd.read_csv(config_util.get(section='Path',key='UserActionDay'))
    user_action_day = user_action_day[user_action_day['time_day'] < end_time]
    user_action_day = user_action_day.drop('time_day',axis=1)
    user.index = user['user_id']
    return user_action_day.join(user,on="user_id",how="outer")



def __get_item_profile(end_time):
    """get item profile from statistics and basic item profile
    """
    print "getting the basic item statistics "
    item = pd.read_csv(config_util.get(section='Path',key='Product'))
    item_action_day = pd.read_csv(config_util.get(section='Path',key='ItemActionDay'))
    item_action_day = item_action_day[item_action_day['time_day'] < end_time]
    item_action_day = item_action_day.drop('time_day',axis=1)
    item.index = item['sku_id']

    return item_action_day.join(item,on="sku_id",how="outer")


def __get_label(end_time,pair):
    """get label to the <user_id,sku_id>
    """



def get_data(split_size = 0.2, granularity = 0, end_time = '2016-04-10'):
    """provide data

     Provide the train_data, train_target , test_data and test_target

    Args:
        split_size: <int>, between 0 and 1, indicate the test proportion.
        granularity: <int-enum>,0 indicates day, 1 indicates week, and 2 indicates month.
        end_time: <date>, the end date of the provided data

    Returns:
        train_data: <dataframe>, A dataframe of train_data, contains all the corresponding feature.
        train_target: <list>, A list of the y_true, the target of train_data
        test_data: <dataframe>, A dataframe of test_data, contains all the corresponding feature.
        test_target: <list>, A list of the y_true, the target of test_data

    the type of target or label is int now, the value of label is determined by statistics of following shoping data.
    TODO: the probability of the label value, to differentiate diversity user

    """
    user_feature = __get_user_profile(end_time)
    item_feature = __get_item_profile(end_time)
    df = pd.concat([user_feature,item_feature],ignore_index=True)
    label = __get_label(end_time,df[['user_id','sku_id']])
    df = df.fillna(0).replace('None', 0).dropna(axis=0, how='all')






