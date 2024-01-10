import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot



st.header('An overview of happiness', divider='rainbow')
st.image('balloons_1.jpeg')
st.sidebar.title('Content')

pages = ['Introduction', 'DataViz', 'Results']
page = st.sidebar.radio('Go to', pages)
df = pd.read_csv('happy_filtered.csv', index_col=0)


if page == pages[0]:
    st.write('The :orange[World Happiness Report] has been published for over 10 years. The way in which happiness is measured has evolved since then. In this project, we evaluated the different relationships between the individual factors. We also wanted to determine the Happiness Score in a world without Covid and compare it with the actual results from the pandemic years.')
    st.divider()
    st.markdown('Dataframe')
    st.dataframe(df)

    unique_countries = df['Country_name'].unique()
    unique_years = df['year'].unique()
    selected_countries = st.multiselect('Select a country:', unique_countries)
    selected_years = st.multiselect('Select a year:', unique_years, default=unique_years)
    if selected_countries and selected_years:
            data_subset = df[df['Country_name'].isin(selected_countries) & df['year'].isin(selected_years)] 
            st.dataframe(data_subset)
    

    
elif page == pages[1]:
    st.markdown('### Data Vizualisation')
    st.markdown('##### Heatmap')

    happy_filtered_heatmap = df.drop(['year', 'Country_name','Regional_indicator'], axis=1)
    fig = px.imshow(happy_filtered_heatmap.corr(), color_continuous_scale='darkmint', text_auto=True, aspect='auto')
    st.plotly_chart(fig, use_container_width=True)    

    st.divider()

    st.markdown('##### Statistical Tests')
    code = '''from scipy.stats import pearsonr
    pearsonr(x = happy_filtered['Life_Ladder'], y = happy_filtered['Log_GDP_per_capita'])
    
    pearsonr(x = happy_filtered['Life_Ladder'], y = happy_filtered['Social_support'])'''
    st.code(code, language='python')
    
    #fig1 = qqplot(happy_filtered_heatmap['Life_Ladder'], line='s').gca().lines
    #st.pyplot(fig1)

    st.divider()
    st.markdown('##### Analysis')
    st.markdown('###### Life Ladder Distribution')
    #selected_years_hist = [2017, 2019, 2021]
    #df_hist = df[df['year'].astype(int).isin(selected_years_hist)].reset_index(drop=True)
    fig_1 = px.histogram(df, x="Life_Ladder", nbins=10, opacity=0.8, color_discrete_sequence=['indianred'], hover_data=df.columns, animation_frame='year')
    fig_1.update_layout(bargap=0.2, bargroupgap=0.1, barmode='group')
    st.plotly_chart(fig_1, use_container_width=True)    

    st.markdown('###### Countries per Regional Indicator')
    happy_filtered_2021 = df.loc[df['year'] == 2021]
    happy_filtered_2021_reg = happy_filtered_2021[['Regional_indicator', 'Country_name']]
    fig_2 = px.bar(happy_filtered_2021_reg, x="Regional_indicator", color='Regional_indicator',
                   hover_data="Country_name")
    fig_2.update_layout(bargap=0.2, bargroupgap=0.1)
    st.plotly_chart(fig_2, use_container_width=True)


    st.markdown('###### Distribution of Life Ladder per Regional Indicator')
    fig_3 = px.box(df, x="Regional_indicator", y="Life_Ladder", color="Regional_indicator", hover_data="Country_name", animation_frame="year")
    st.plotly_chart(fig_3, use_container_width=True, showlegend=False)


    st.markdown('###### Distribution of Generosity per Regional Indicator')
    fig_4 = go.Figure()
    fig_4.add_traces(go.Box(x=df['Regional_indicator'],
                      y=df['Generosity']))
    st.plotly_chart(fig_4, use_container_width=True)


    st.markdown('###### GDP per Regional Indicator over the years')
    fig_5= px.scatter(df, x="Log_GDP_per_capita", y="Life_Ladder", color="Regional_indicator", hover_data=['Country_name'], animation_frame="year")
    st.plotly_chart(fig_5, use_container_width=True)
    

 #st.dataframe(df.head())
    #selected_years_plot = [2015, 2016, 2017, 2018, 2019]
    #df = df.groupby(by =['Regional_indicator', 'year'])['Life_Ladder'].mean().reset_index()
    #df_recent = df[df['year'].astype(int).isin(selected_years_plot)].reset_index(drop=True)
    #st.line_chart(data= df_recent, x='year', y='Life_Ladder', color='Regional_indicator', use_container_width=True)

#uploads = st.file_uploader("Select files ", type=['csv', 'CSV', 'xlsx'],accept_multiple_files=False)

#if uploads is not None:
#        my_dataset = load_csv(uploads)
#        my_dataset = my_dataset.drop(['level_happiness'], axis=1)    
#        print("Got the uploads!!!")
#        
#        st.dataframe(my_dataset, use_container_width=True, column_order=('Country_name', 'Regional_indicator', 'year', 'Life_Ladder', 'Log_GDP_per_capita', 'Social_support', 'Healthy_life_expectanca_at_birth', 'Freedom_to_make_life_choices', 'Generosity', 'Perceptions_of_corruption'))