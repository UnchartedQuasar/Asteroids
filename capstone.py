import numpy as np
import requests
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
from time import perf_counter
#import psycopg2


tick = perf_counter()
st.title('Asteroids')
st.write('Date range = 7 days')
start_date = st.date_input('Enter start date',
                           value = date.today() - timedelta(days=7),
                           min_value = date(1899,12,30),
                           max_value = date.today() - timedelta(days=7))
end_date = start_date + timedelta(days=7)
if end_date > date.today():
    raise ValueError('End date cannot be in the future')
st.write(f'End date: {end_date}')

#today = str(end_date)
#start_date = end_date - timedelta(days=7)
api_key = 'JwvyuW8z5oyap0g2gNCLdHzikMvQGgXlwsc5a8pQ'
url = f'https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}'

asteroids = requests.get(url).json()

def get_info(ast_num, date):
  global url
  global asteroids
  asteroid_deets = asteroids['near_earth_objects'][str(date)][ast_num]
  id_ = asteroid_deets['id']
  velocity = float(asteroid_deets['close_approach_data'][0]['relative_velocity']['kilometers_per_second'])
  diameters = list(asteroid_deets['estimated_diameter']['kilometers'].values())
  avg_diam = float(sum(diameters)/2)
  log_d = float(np.log10(avg_diam))
  abs_mag = float(asteroid_deets['absolute_magnitude_h'])
  miss_dist = float(asteroid_deets['close_approach_data'][0]['miss_distance']['lunar'])
  hazardous = asteroid_deets['is_potentially_hazardous_asteroid']


  return id_, velocity, avg_diam, log_d, abs_mag, miss_dist, hazardous


asteroid_df = pd.DataFrame([], columns=['id_', 'velocity(km/s)', 'avg_diameter(km)', 'log(diameter)', 'abs_magnitude', 'miss_distance(LD)', 'potential_hazard'])

date_list = [end_date - timedelta(days=x) for x in range(8)]
for date_ in date_list:
  for i in range(len(asteroids['near_earth_objects'][str(date_)])):
    id_,   velocity,   avg_diameter,   log_d,   abs_magnitude,   miss_distance,   potential_hazard = (get_info(i, str(date_)))
    asteroid_df.loc[len(asteroid_df.index)] = [id_,   velocity,   avg_diameter,  log_d,   abs_magnitude,   miss_distance,   potential_hazard]
m,c = np.polyfit(asteroid_df['log(diameter)'], asteroid_df['abs_magnitude'], 1)
asteroid_df['log_d_H_m'] = asteroid_df['log(diameter)']*m+c
'''
conn = psycopg2.connect(database="pagila",
        user = "de_vich",
        password = "cri",
        host = "data-sandbox.c1tykfvfhpit.eu-west-2.rds.amazonaws.com",
        port = "5432"  # Default PostgreSQL port
    )

def get_something_from_sql(sql_command):
    try:    
        with conn.cursor() as cur:
            cur.execute(sql_command)

            result = cur.fetchone()
            if result[0]:
                return result
            else:
                return None

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error:", error)
        return None

def update_table(sql_command):
    for index, row in asteroid_df.iterrows():
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    {sql_command}
                """ % (row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7]))
    
                conn.commit()  # Commit the changes
    
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error:", error)

#sql = get_something_from_sql("select * from student.vc_neurogym vn LIMIT 50")
update_table("insert into student.vc_asteroids values ('%s',%s,%s,%s,%s,%s,%s,%s);")
'''

haz_graph = asteroid_df.groupby('potential_hazard').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.xlabel('Frequency')
plt.ylabel('Potential hazard')
plt.title('Number of asteroids that are potentially hazardous')
st.pyplot(haz_graph.figure)
plt.show()
plt.clf()


d_H_graph = asteroid_df.plot(kind='scatter', x='avg_diameter(km)', y='abs_magnitude', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.xlabel('Average diameter (km)')
plt.ylabel('H')
plt.title('Absolute magnitude against asteroid diameter')
st.pyplot(d_H_graph.figure)
plt.show()
plt.clf()


log_d_H_graph = plt.scatter(x=asteroid_df['log(diameter)'], y=asteroid_df['abs_magnitude'], s=32, alpha=.8)
plt.plot(asteroid_df['log(diameter)'], asteroid_df['log_d_H_m'])
plt.xlabel('log(d)')
plt.ylabel('H')
plt.title('Absolute magnitude (H) against log(diameter)')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.text(-1.1, 28.5, 'H = ' + ' {:.2f}'.format(m) + 'log(d)' + ' + {:.2f}'.format(c), size=14)
st.pyplot(log_d_H_graph.figure)
plt.show()
plt.clf()


tock = perf_counter()
print(round(tock-tick, 2))
