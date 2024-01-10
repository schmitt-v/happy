import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.gofplots import qqplot
from PIL import Image

st.header('The Pursuit of Happinness', divider='rainbow')
st.sidebar.title('Content')

pages = ['Introduction', 'Data Vizualisation - Part 1', 'Data Vizualisation - Part 2', 'Modeling']
page = st.sidebar.radio('Go to', pages)
df = pd.read_csv('happy_filtered_final_2.csv', index_col=0)
df_avg = pd.read_csv('predictions_moving_average.csv', index_col=0)
df_avg_20 = pd.read_csv('predictions_moving_average_2020_2021.csv', index_col =0)
df_mae = pd.read_csv('mae_happy_17_21.csv', index_col=0)
df_mae_1 = pd.read_csv('mae_happy_20_21.csv')
df_avg_mae = pd.read_csv('mae_happy_avg_17_21.csv')
merged_variables = pd.read_csv('pred_social_gdp_health_17_21.csv')

if page == pages[0]:
    st.image('balloons_1.jpeg')
    st.write("The :orange[World Happiness Report] has been published by the Sustainable Development Solutions Network for over 10 years. The evaluation of happiness has evolved since then and more and more factors were included to find out the country where people feel most happy.")
    st.divider()

    st.write("The most important evaluation metrics are:")
    expander = st.expander('- :orange[Life_Ladder:]')
    expander.write('National average answers to the question "Please imagine a ladder, with steps numbered from 0 at the bottom to 10 at the top. The top of the ladder represents the best possible life for you and the bottom of the ladder represents the worst possible life for you. On which step of the ladder would you say you personally feel you stand at this time?"')	
    
    expander_1 = st.expander('- :orange[Log_GDP_per_capita:]')
    expander_1.write('The statistics of GDP per capita in purchasing power parity at constant. Taken from World Development Indicators (WDI).')	

    expander_2 = st.expander('- :orange[Healthy_life_expectancy_at_birth:]')
    expander_2.write('Healthy life expectancies at birth are based on the data extracted from the World Health Organization’s (WHO) Global Health Observatory')	

    expander_3 = st.expander('- :orange[Social_support:]')
    expander_3.write('National average of the binary responses [either 0 for "no" or 1 for "yes"] to the question "If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?"')	

    expander_4 = st.expander('- :orange[Freedom_tom_make_life_choices:]')
    expander_4.write('National average of binary responses for the question "Are you satisfied or dissatisfied with your freedom to choose what you do with your life?"')	

    expander_5 = st.expander('- :orange[Generosity:]')
    expander_5.write('National average of response to the GWP question "Have you donated money to a charity in the past month?"')	

    expander_6 = st.expander('- :orange[Perceptions_of_corruption:]')
    expander_6.write('National average of binary responses for the questions "Is corruption widespread throughout the government or not?" and "Is corruption widespread within businesses or not?"')	


    st.markdown('Dataframe')
    st.dataframe(df)

    unique_countries = df['Country_name'].unique()
    unique_years = df['year'].unique()
    selected_countries = st.multiselect('Select a country:', unique_countries)
    selected_years = st.multiselect('Select a year:', unique_years, default=unique_years)
    if selected_countries and selected_years:
            data_subset = df[df['Country_name'].isin(selected_countries) & df['year'].isin(selected_years)] 
            st.dataframe(data_subset)
    
    st.divider()
    st.markdown("##### Project's objective")
    st.write("Our goal was to evaluate the different relationships between the individual factors and to investigate if there's a significant difference in Happiness Scores without the impact of the pandemic.")
    
    

    
elif page == pages[1]:
    st.image('balloons_1.jpeg')
    st.markdown('### Visual Exploration - Part 1')
    st.markdown('##### Heatmap - the correlation coefficients')

    happy_filtered_heatmap = df.drop(['year', 'Country_name','Regional_indicator', 'level_happiness'], axis=1)
    fig = px.imshow(happy_filtered_heatmap.corr(), color_continuous_scale='darkmint', text_auto=True, aspect='auto')
    st.plotly_chart(fig, use_container_width=True)    

    st.caption("The first step was to get familiar with the possible correlations in the dataset.")
    st.caption('Especially interesting were the correlations of the column "Life_Ladder" containing the Happiness-Ladder-score with the other variables of the data set like "Log_GDP_per_capita" or "Healthy_life_expectancy_at_birth".')
             
    st.divider()

    st.markdown('#### Statistical Tests')
    st.write("The correlation coefficients of the heatmap were also checked by running a statistical test (Pearson test).")
    code = '''from scipy.stats import pearsonr
    pearsonr(x = happy_filtered['Life_Ladder'], y = happy_filtered['Log_GDP_per_capita'])
    
    pearsonr(x = happy_filtered['Life_Ladder'], y = happy_filtered['Social_support'])'''
    st.code(code, language='python')

    st.write('Output: PearsonRResult(statistic=0.7831064193402993, pvalue=0.0)')
    st.caption('The results of the Pearson test done for all variables confirmed all correlation coefficients in type and intensity as depicted in the heatmap.')

    st.divider()
    st.markdown('#### Distribution of Life Ladder')
    fig_3 = px.histogram(df, x="Life_Ladder", nbins=8, opacity=0.8, color_discrete_sequence=['cornflowerblue'],
                   animation_frame="year")
    fig_3.update_layout(bargap=0.2, bargroupgap=0.1)
    st.plotly_chart(fig_3)
    
    st.markdown('#### qqplot Healthy life expectancy')
    qq_plot = qqplot(happy_filtered_heatmap['Healthy_life_expectancy_at_birth'], line='s', fit=True, marker='o', color='blue')

    x, y = qq_plot.gca().lines[0].get_xydata().T

    fig_2 = px.scatter(x=x, y=y, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                 template='plotly_white')

    fig_2.add_shape(type='line', x0=min(x), y0=min(x), x1=max(x), y1=max(x), line=dict(dash='dash', color='white'))
    st.plotly_chart(fig_2)
    st.caption('The distribution of the values for the column Healthy_life_expectancy_at_birth do not follow a normal distribution as the other values of the dataset.')
    
    #st.markdown('#### qqplot Life Ladder')
    #qq_plot_1 = qqplot(happy_filtered_heatmap['Life_Ladder'], line='s', fit=True, marker='o', color='blue')

    #x_1, y_1 = qq_plot_1.gca().lines[0].get_xydata().T

    #fig_3 = px.scatter(x=x_1, y=y_1, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
    #             template='plotly_white')

    #fig_3.add_shape(type='line', x0=min(x_1), y0=min(x_1), x1=max(x_1), y1=max(x_1), line=dict(dash='dash', color='white'))
    #st.plotly_chart(fig_3)




elif page == pages[2]:
    st.image('balloons_1.jpeg')
    st.markdown('### Visual Exploration - Part 2')
    st.markdown('#### Analysis')

    st.markdown('##### Countries per Regional Indicator')
    happy_filtered_2021 = df.loc[df['year'] == 2021]
    region_counts = happy_filtered_2021['Regional_indicator'].value_counts().reset_index()
    region_counts.columns = ['Regional_indicator', 'Count']
    happy_filtered_2021_reg = pd.merge(happy_filtered_2021[['Regional_indicator', 'Country_name']], region_counts, on='Regional_indicator')
    happy_filtered_2021_reg = happy_filtered_2021_reg.sort_values(by='Count', ascending=True)
    fig_2 = px.bar(happy_filtered_2021_reg, x='Regional_indicator', color='Regional_indicator',
               hover_data="Country_name", labels={'Regional_indicator': 'Region', 'Count': 'Anzahl'})
    fig_2.update_layout(bargap=0.2, bargroupgap=0.1, xaxis={'visible': False})

    st.plotly_chart(fig_2, use_container_width=True)
    st.caption('To make the visual exploration more accessable we used the Regional_indicator to sort all 149 countries into 10 Regional groups.')
    

    st.markdown('##### Distribution of Life Ladder per Regional Indicator')
    df_sorted = df.sort_values(by=['Regional_indicator', 'Life_Ladder'])
    fig_3 = px.box(df_sorted, x="Regional_indicator", y="Life_Ladder", color="Regional_indicator", hover_data="Country_name", animation_frame="year")
    fig_3.update_layout(xaxis={'visible': False})
    st.plotly_chart(fig_3, use_container_width=True)

    st.markdown('##### Distribution of Life Ladder per Regional Indicator')
    fig_7 = px.box(df, x="Regional_indicator", y="Life_Ladder", color="level_happiness", hover_data="Regional_indicator", animation_frame="year")
    fig_7.update_layout(xaxis={'visible': False})
    st.plotly_chart(fig_7, use_container_width=True)
    st.caption('We notice that there is not much movement between the single regions throughout the years. Three groups of regions can be identified: one group with a very high level of happiness, one big middle group with a mid-level of happiness and one group with a very low average lader score and therefore with a low level of happiness.')

    #st.markdown('##### Distribution of Generosity per Regional Indicator')
    #df_sorted = df.sort_values(by=['Regional_indicator', 'Generosity'])
    #fig_4 = px.box(df, x="Regional_indicator", y="Generosity", color="level_happiness", hover_data="Country_name", animation_frame="year")
    #fig_4.update_layout(xaxis={'visible': False})
    #st.plotly_chart(fig_4, use_container_width=True)
    #st.caption('Interestingly the area of the world where generosity seems to be most present is Souteast Asia. Nonetheless the ladder scores of this remain in the mid-level happiness group. But this is not a surprise as according to the heatmap there is no correlation between Generosity and the Ladder score.')


    st.markdown('##### The bigger the economy the happier?')
    fig_5= px.scatter(df, x="Log_GDP_per_capita", y="Life_Ladder", color="level_happiness", hover_data=['Country_name', 'Regional_indicator'], animation_frame="year")
    st.plotly_chart(fig_5, use_container_width=True)
    st.caption('The scatter plot shows a very high and significant correlation between the GDP per capita of a country and the Ladder score. Yet there are exceptions to the rule and some countries like "Botswana" with a relatively high GDP have a very low Ladder score.')


    st.markdown('##### Evolution of the GDP an Life Expectancy')
    fig_6 = px.scatter(df, x="Healthy_life_expectancy_at_birth", y="Log_GDP_per_capita", color="level_happiness", hover_data=['Country_name', 'Regional_indicator'], animation_frame="year")
    st.plotly_chart(fig_6, use_container_width=True)
    st.caption('The strongest positive correlation of the dataset is between the GDP and the Life expectancy. For years both factors were on the rise - with the year 2020 and the breakout of COVID this development stopped.')


elif page == pages[3]:
    st.markdown('### Modeling')

    image = Image.open('Covid Mood Picture.jpg')
    st.image(image, caption = 'A view of the world during COVID, ©unsplash.com')
    st.write("The Project's objective was to train a model to predict the values of the World Happiness Report for the years 2020 and 2021 based on the values for the years 2009 - 2019 and thereby predict the values for a 'world without COVID'.")

    expander_1 = st.expander(':orange[Modeling Approach]')
    expander_1.write("As we were dealing with a multi-dimensional regression problem (we are predicting numerical values with almost infinite possibilites for several target values) linked to a time dimension, finding the right modeling approach was crucial to achieve evaluable results.")
    expander_1.write("Our approach was to first use a 'moving average'-function to predict values for the years 2020 and 2021 for all countries and their variables based on the five previous values of a variable." 
                     " In a next step we trained a model to predict a defined target value (e.g. Life_Ladder). The training set consisted of the reported values for the years 2009 - 2019. For prediction we used the values provided by the 'moving average'-function.")

    expander_2 = st.expander(':orange[Choice of Model]')
    expander_2.write("To model the regression problem we tried three algorithms:")
    expander_2.write("- Linear Regression")
    expander_2.write("- XG Boost")
    expander_2.write("- RandomForest")
    expander_2.write("The MSE-metric of all algorithms ranged between 0.27 (Linear Regression) and 0.12 (RandomForest). However the decision tree-models (RandomForest and XG Boost in dftree- and dart-mode) showed a significant overfitting. Only the linear models were robust to this problem and showed no signs of overfitting on the given dataset.")
    expander_2.write("Consequently we sticked with the Linear Regression Model as it showed no signs of overfitting and the already low MSE-metric of 0.27 showed a high precision of the model to calculate the needed predictions.")
    expander_2.write("A further hypertuning of the Linear Regression algorithm using a Grid Search took place as well and we earned a final MSE-metric of 0.25.")
 
    st.divider()
    st.markdown('#### Predictions Linear Regression')

    st.markdown('##### Dataframe')
    st.dataframe(df_avg)

    #st.divider()

    #st.markdown('###### The MAE per Regional Indicator')

    #grouped_mae = df_mae_1.groupby('Country_name').agg({'Life_Ladder': 'mean', 'Life_Ladder_Avg': 'mean', 'MAE': 'mean'}).reset_index()

    #fig_4 = px.scatter(grouped_mae, x='Life_Ladder', y='Life_Ladder_Avg', size='MAE',
    #                  hover_data=['Country_name', 'MAE'], opacity=0.6)

    #st.plotly_chart(fig_4, use_container_width=True)

    st.divider()
    st.markdown('##### Evaluation')
    st.markdown('###### Life Ladder Predictions')
   
    grouped_df_avg = df_mae.groupby(['Country_name', 'year']).agg({'Life_Ladder': 'mean', 'Life_Ladder_Avg': 'mean'}).reset_index()
    selected_countries = ['Ethiopia', 'Germany', 'France', 'India', 'Morocco']
    grouped_avg_plot = grouped_df_avg[grouped_df_avg['Country_name'].isin(selected_countries)].reset_index(drop=True)
    fig= go.Figure()

    for region in grouped_avg_plot['Country_name'].unique():
        region_data = grouped_avg_plot[grouped_avg_plot['Country_name'] == region]

        trace1 = go.Scatter(x=region_data['year'], y=region_data['Life_Ladder'], mode='lines+markers', name=f'Life Ladder - {region}')
        fig.add_trace(trace1)

        trace2 = go.Scatter(x=region_data['year'], y=region_data['Life_Ladder_Avg'], mode='lines+markers', name=f'Life Ladder pred - {region}', line=dict(dash='dash'))
        fig.add_trace(trace2)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Seemingly the Ladder score of most countries measuring the happiness was not negatively affected. Actually some countries show higher values compared to the models' predictions.")
    st.caption("Nonetheless we can also see in the example of India that the models' predictions are way higher than the reported values. In this case the outbreak of COVID might have strongly affected the countries' happiness.")
    
    #st.markdown('###### Life Ladder (MOV AVG)')
    #grouped_plot = df_avg_mae.groupby(['Country_name', 'year']).agg({'Life_Ladder': 'mean', 'Life_Ladder_Avg': 'mean'}).reset_index()
    #selected_countries = ['Colombia', 'Ethiopia', 'France', 'Germany', 'Morocco', 'India']
    #df_avg_plot = grouped_plot[grouped_plot['Country_name'].isin(selected_countries)].reset_index(drop=True)
    #fig_2= go.Figure()

    #for region in df_avg_plot['Country_name'].unique():
    #    region_data = df_avg_plot[df_avg_plot['Country_name'] == region]

    #    trace1 = go.Scatter(x=region_data['year'], y=region_data['Life_Ladder'], mode='lines+markers', name=f'Life Ladder - {region}')
    #    fig_2.add_trace(trace1)

    #    trace2 = go.Scatter(x=region_data['year'], y=region_data['Life_Ladder_Avg'], mode='lines+markers', name=f'Life Ladder pred - {region}', line=dict(dash='dash'))
    #    fig_2.add_trace(trace2)

    #st.plotly_chart(fig_2, use_container_width=True)
    st.divider()
    st.markdown('###### GDP Predictions')
    merged_gdp = merged_variables.groupby(['Country_name', 'year']).agg({'Log_GDP_per_capita': 'mean', 'Log_GDP_per_capita_pred': 'mean'}).reset_index()
    selected_countries = ['Ethiopia', 'Germany', 'France', 'India', 'Morocco']
    merged_gdp_group = merged_gdp[merged_gdp['Country_name'].isin(selected_countries)].reset_index(drop=True)
    fig_3= go.Figure()

    for region in merged_gdp_group['Country_name'].unique():
        region_data = merged_gdp_group[merged_gdp_group['Country_name'] == region]

        trace1 = go.Scatter(x=region_data['year'], y=region_data['Log_GDP_per_capita'], mode='lines+markers', name=f'GDP - {region}')
        fig_3.add_trace(trace1)

        trace2 = go.Scatter(x=region_data['year'], y=region_data['Log_GDP_per_capita_pred'], mode='lines+markers', name=f'GDP pred - {region}', line=dict(dash='dash'))
        fig_3.add_trace(trace2)
    st.plotly_chart(fig_3, use_container_width=True)

    st.caption("For almost all countries we can see that the reported values are below the models' predictions. Depending on the individual country this difference can be very significant. This might be a sign that most countries were facing the impact of the COVID-crisis on an economical level.")
    st.caption("However some countries like India seemingly show no direct signs of a negative economical impact.")

    st.markdown('###### Life Expectancy Predictions')
    merged_health = merged_variables.groupby(['Country_name', 'year']).agg({'Healthy_life_expectancy_at_birth': 'mean', 'Healthy_life_expectancy_at_birth_pred': 'mean'}).reset_index()
    selected_countries = ['Ethiopia', 'Germany', 'France', 'India', 'Morocco']
    merged_health_group = merged_health[merged_health['Country_name'].isin(selected_countries)].reset_index(drop=True)
    fig_6= go.Figure()

    for region in merged_health_group['Country_name'].unique():
        region_data = merged_health_group[merged_health_group['Country_name'] == region]

        trace1 = go.Scatter(x=region_data['year'], y=region_data['Healthy_life_expectancy_at_birth'], mode='lines+markers', name=f'Life Expectancy - {region}')
        fig_6.add_trace(trace1)

        trace2 = go.Scatter(x=region_data['year'], y=region_data['Healthy_life_expectancy_at_birth_pred'], mode='lines+markers', name=f'Life Expectancy pred - {region}', line=dict(dash='dash'))
        fig_6.add_trace(trace2)
    st.plotly_chart(fig_6, use_container_width=True)

    st.caption("Some countries show a possible negative impact of COVID on their life expectancy. The models' predictions are higher than the reported values.")
    st.caption("However a lot countries seemingly show no decline in their life expectancy. The value remained stable through the COVID years.")

    #st.markdown('###### Social Support (LIN REG)')
    #merged_social = merged_variables.groupby(['Country_name', 'year']).agg({'Social_support': 'mean', 'Social_support_pred': 'mean'}).reset_index()
    #selected_countries = ['Colombia', 'Ethiopia', 'France', 'Germany', 'Morocco', 'India']
    #merged_social_group = merged_social[merged_social['Country_name'].isin(selected_countries)].reset_index(drop=True)
    #fig_7= go.Figure()

    #for region in merged_social_group['Country_name'].unique():
    #    region_data = merged_social_group[merged_social_group['Country_name'] == region]

    #    trace1 = go.Scatter(x=region_data['year'], y=region_data['Social_support'], mode='lines+markers', name=f'Social Support - {region}')
    #    fig_7.add_trace(trace1)

    #    trace2 = go.Scatter(x=region_data['year'], y=region_data['Social_support_pred'], mode='lines+markers', name=f'Social Support pred - {region}', line=dict(dash='dash'))
    #    fig_7.add_trace(trace2)
    #st.plotly_chart(fig_7, use_container_width=True)

    #st.divider()

    #st.markdown('###### Mean Absolute Error')
    #st.dataframe(df_mae_1)

    #unique_countries = df_mae_1['Country_name'].unique()
    #unique_years = df_mae_1['year'].unique()
    #selected_countries = st.multiselect('Select a country:', unique_countries)
    #selected_years = st.multiselect('Select a year:', unique_years, default=unique_years)
    #if selected_countries and selected_years:
    #        data_subset = df_mae_1[df_mae_1['Country_name'].isin(selected_countries) & df_mae_1['year'].isin(selected_years)] 
    #        st.dataframe(data_subset)

    st.divider()

    expander = st.expander(':orange[Conclusion and Outlook]')
    expander.write("After a first look at the data and the predictions we see that the COVID-crisis did not affect all countries in the same way.")
    expander.write("The value that showed the fewest signs of possibly being negatively affected by COVID was actually the Ladder score measuring the happiness of a country. Seemingly COVID had not a devastating effect on worlds' happiness in general. ")
    expander.write("However the data gave us signs that COVID had a significant negative impact especially on the GDP of most countries as well as on the life expectancy in some cases.")
    expander.write("To truly link the results in the data to COVID, further investigation is needed to include data on COVID (number of COVID-related deaths per population) in the data framework and to investigate whether the countries that experienced a sharp decline in GDP, for example, were actually severely affected by the pandemic.")
    expander.write("Furthermore, in a next step, we would use a model that takes time into account by implementing a time series and comparing it to our previous results. Other possible steps include implementing additional features that define dependencies to account for the influence of certain factors such as a pandemic, natural disasters or wars.")


    #grouped_countries = df_mae.groupby(['Country_name', 'year', 'Regional_indicator']).agg({'Life_Ladder': 'mean', 'Life_Ladder_Avg': 'mean', 'MAE': 'mean'}).reset_index()
    #selected_countries = ['Colombia', 'Ethiopia', 'France', 'Germany', 'Morocco', 'Japan']
    #grouped_mae_plot = grouped_countries[grouped_countries['Country_name'].isin(selected_countries)].reset_index(drop=True)

    #fig_5 = px.scatter(grouped_mae_plot, x='Life_Ladder', y='Life_Ladder_Avg', color='Country_name', size='MAE',
    #                  hover_data=['Country_name', 'Regional_indicator', 'year'], opacity=0.6)

    #st.plotly_chart(fig_5, use_container_width=True)



    

    #selected_years_ = [2020, 2021]
    #df_plot = df[df['year'].astype(int).isin(selected_years_hist)].reset_index(drop=True)


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