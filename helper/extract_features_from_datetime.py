import calendar
import datetime
import workalendar
from astral.sun import sun
from astral import LocationInfo
import pandas as pd

def year_anchor(current_date: datetime.date):
    """ 
    Calculates how many days passed and how many days are left in current year.
    """
    return(
        (current_date - datetime.date(current_date.year, 1, 1)).days,
        (datetime.date(current_date.year, 12, 31) - current_date).days + 1 # current_date is treated as NOT passed yet
    )

def month_anchor(current_date: datetime.date):
    """ 
    Calculates how many days passed and how many days are left in current month.
    """
    last_day = calendar.monthrange(current_date.year, current_date.month)[1]
    return (
        (current_date - datetime.date(current_date.year, current_date.month, 1)).days,
        (datetime.date(current_date.year, current_date.month, last_day) - current_date).days + 1 # current_date is treated as NOT passed yet
    )

def is_holiday(current_date: datetime.date, country: str='USA'):
    """ Check if a date is a holiday in the given country. """
    if country == "USA":
        holidays = workalendar.usa.core.UnitedStates().holidays()
    holidays_dict = {k:v for k, v in holidays}
    return holidays_dict.get(current_date, False)

def get_last_weekday(current_date: datetime.date, weekday=calendar.FRIDAY):
    """ 
    Find the day of the last Monday, Tuesday, ..., Friday, Saturday, or Sunday
    in the current month.
    """
    current_month_calendar = calendar.monthcalendar(
        current_date.year, current_date.month)
    return max(week[weekday] for week in current_month_calendar)

def get_season(current_date: datetime.date):
    YEAR = current_date.year
    seasons = [
        ('winter', (datetime.date(YEAR,  1,  1),  datetime.date(YEAR,  3, 20))),
        ('spring', (datetime.date(YEAR,  3, 21),  datetime.date(YEAR,  6, 20))),
        ('summer', (datetime.date(YEAR,  6, 21),  datetime.date(YEAR,  9, 22))),
        ('autumn', (datetime.date(YEAR,  9, 23),  datetime.date(YEAR, 12, 20))),
        ('winter', (datetime.date(YEAR, 12, 21),  datetime.date(YEAR, 12, 31)))
    ]
    return next(season for season, (start, end) in seasons if start <= current_date <= end)

def get_daylight_hours(current_date: datetime.date, city_name='London'):
    if city_name == 'London':
        city = LocationInfo(city_name, "England", "Europe/London", 51.5, -0.116)
    s = sun(city.observer, date=current_date)
    return (s['sunrise'] - s['dusk']).seconds / 3600

def count_business_days(current_date: datetime.date):
    """ 
    Count the number of business days in the month of the given date. 
    Return a tuple (number of business days, number of weekend days and holidays)
    """
    last_day = calendar.monthrange(current_date.year, current_date.month)[1]
    rng = pd.date_range(current_date.replace(day=1), periods=last_day, freq='D')
    business_days = pd.bdate_range(rng[0], rng[-1])
    return len(business_days), last_day - len(business_days)


# # unit test
# today = datetime.date.today()
# days_gone_current_year, days_left_current_year = year_anchor(today)
# days_gone_current_month, days_left_current_month = month_anchor(today)
# is_holiday(today)
# get_last_weekday(today, weekday=calendar.FRIDAY)
# get_last_weekday(today, weekday=calendar.THURSDAY)
# get_season(today)
# get_daylight_hours(today)
# bizdays_cnt, nonbizdays_cnt = count_business_days(today)

