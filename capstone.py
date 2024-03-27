import numpy as np
import requests
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
import psycopg2


st.title('Asteroids')
end_date = date.today()
start_date = end_date - timedelta(days=0)
st.write(f'Date range : 2024-03-20 - {end_date}')

api_key = 'JwvyuW8z5oyap0g2gNCLdHzikMvQGgXlwsc5a8pQ'
url = f'https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}'

asteroids = requests.get(url).json()

def get_info(ast_num, date_):
  global url
  global asteroids
  asteroid_deets = asteroids['near_earth_objects'][str(date_)][ast_num]
  id_ = asteroid_deets['id']
  velocity = float(asteroid_deets['close_approach_data'][0]['relative_velocity']['kilometers_per_second'])
  diameters = list(asteroid_deets['estimated_diameter']['kilometers'].values())
  avg_diam = float(sum(diameters)/2)
  log_d = float(np.log10(avg_diam))
  abs_mag = float(asteroid_deets['absolute_magnitude_h'])
  miss_dist = float(asteroid_deets['close_approach_data'][0]['miss_distance']['lunar'])
  hazardous = asteroid_deets['is_potentially_hazardous_asteroid']


  return id_, velocity, avg_diam, log_d, abs_mag, miss_dist, hazardous


asteroid_df = pd.DataFrame([], columns=['id_','date_', 'velocity(km/s)', 'avg_diameter(km)', 'log(diameter)', 'abs_magnitude', 'miss_distance(LD)', 'potential_hazard'])

date_list = [end_date - timedelta(days=x) for x in range(1)]
for date_ in date_list:
  for i in range(len(asteroids['near_earth_objects'][str(date_)])):
    id_,   velocity,   avg_diameter,   log_d,   abs_magnitude,   miss_distance,   potential_hazard = (get_info(i, str(date_)))
    asteroid_df.loc[len(asteroid_df.index)] = [id_, str(date_),   velocity,   avg_diameter,  log_d,   abs_magnitude,   miss_distance,   potential_hazard]
m,c = np.polyfit(asteroid_df['log(diameter)'], asteroid_df['abs_magnitude'], 1)
asteroid_df['log_d_H_m'] = asteroid_df['log(diameter)']*m+c

conn = psycopg2.connect(database="pagila",
        user = "de_vich",
        password = "cri",
        host = "data-sandbox.c1tykfvfhpit.eu-west-2.rds.amazonaws.com",
        port = "5432"  # Default PostgreSQL port
    )

def get_df(sql_command):
    try:    
        with conn.cursor() as cur:
            cur.execute(sql_command)

            result = cur.fetchall()
            if result[0]:
                return result
            else:
                return None

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error:", error)
        return None


def update_table(sql_command, df):
    for index, row in df.iterrows():
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    {sql_command}
                """ % (row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7], row.iloc[8]))
    
                conn.commit()  # Commit the changes
    
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error:", error)

sql = get_df("select * from student.vc_asteroid vn order by date_ desc, id;")
if tuple(asteroid_df.values[0])[0:2]==sql[0][0:2]:
    print('passed away')
else:    
    update_table("insert into student.vc_asteroid values ('%s','%s',%s,%s,%s,%s,%s,%s,%s);", asteroid_df)


sql = [list(sql[i]) for i in range(len(sql))]
sql_df = pd.DataFrame(sql, columns=['id_','date_', 'velocity(km/s)', 'avg_diameter(km)', 'log(diameter)', 'abs_magnitude', 'miss_distance(LD)', 'potential_hazard', 'log_d_H_m'])

haz_nums = get_df('SELECT count(id) AS frequency FROM student.vc_asteroid va GROUP BY potential_hazard;')

tab1, tab2 = st.tabs(['Hazardous Asteroids!!',
                            'Absolute Magnitude vs Diameter'])

haz_graph = sql_df.groupby('potential_hazard').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.text(((haz_nums[0][0])/2)-3, 0, haz_nums[0][0])
plt.text(haz_nums[1][0]+1, 1, haz_nums[1][0])
plt.xlabel('Frequency')
plt.ylabel('Potential hazard')
plt.title('Number of asteroids that are potentially hazardous')
tab1.pyplot(haz_graph.figure)
plt.show()
plt.clf()

figsize = (12, 1.2 * len(sql_df['potential_hazard'].unique()))
plt.figure(figsize=figsize)
plt.title('Hazards comapared to Velocity')
plt.set(xticklabels=[])
haz_v_graph = sns.violinplot(sql_df, x='velocity(km/s)', y='potential_hazard', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
tab1.pyplot(haz_v_graph.figure)

figsize = (12, 1.2 * len(sql_df['potential_hazard'].unique()))
plt.figure(figsize=figsize)
plt.title('Hazards comapared to Diameter')
haz_d_graph = sns.violinplot(sql_df, x='avg_diameter(km)', y='potential_hazard', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
tab1.pyplot(haz_d_graph.figure)

figsize = (12, 1.2 * len(sql_df['potential_hazard'].unique()))
plt.figure(figsize=figsize)
plt.title('Hazards comapared to Miss Distance')
haz_md_graph = sns.violinplot(sql_df, x='miss_distance(LD)', y='potential_hazard', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
tab1.pyplot(haz_md_graph.figure)

with tab2:
    col1, col2 = st.columns(2)
    d_H_graph = sql_df.plot(kind='scatter', x='avg_diameter(km)', y='abs_magnitude', s=32, alpha=.8)
    plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.xlabel('Average diameter (km)')
    plt.ylabel('H')
    plt.title('Absolute magnitude (H) against asteroid diameter')
    col1.pyplot(d_H_graph.figure)
    plt.show()
    plt.clf()
    
    log_d_H_graph = plt.scatter(x=sql_df['log(diameter)'], y=sql_df['abs_magnitude'], s=32, alpha=.8)
    plt.plot(sql_df['log(diameter)'], sql_df['log_d_H_m'])
    plt.xlabel('log(d)')
    plt.ylabel('H')
    plt.title('Absolute magnitude (H) against log(diameter)')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.text(-1.1, 28.5, 'H = ' + ' {:.2f}'.format(m) + 'log(d)' + ' + {:.2f}'.format(c), size=14)
    col2.pyplot(log_d_H_graph.figure)
    plt.show()
    plt.clf()
