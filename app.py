import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.graph_objs as go
import pandas as pd
import plotly.figure_factory as ff
from dash.dependencies import Input, Output, State
import json
import pandas as pd
import plotly_express as px
from datetime import datetime as dt
import dill
import requests
from joblib import Parallel, delayed
import joblib
import numpy as np
# import datetime
from datetime import datetime

external_stylesheets = ['https://codepen.io/plotly/pen/EQZeaW.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('fire_size_map.csv')
df_cause = pd.read_csv('fire_cause.csv')
df_states = pd.read_csv('fire_states.csv')
# rf_joblib = joblib.load('rf_model.pkl')
rf_joblib = joblib.load('random_forest.pkl')
logistic_joblib = joblib.load('logistic.pkl')
svm_joblib = joblib.load('svm.pkl')
gbt_joblib = joblib.load('gbt.pkl')

scaler = joblib.load('standard_scaler.pkl')
size_encoder = joblib.load('size_encoder.pkl')
week_encoder = joblib.load('week_encoder.pkl')
state_encoder = joblib.load('state_encoder.pkl')

fire_size = [ 30, 60, 120, 250, 550, 1500]
fire_count = [ 10, 50, 100, 200, 300, 400]

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]

reds = ['#d1aad2','#d189c2','#dc62ad','#e23990','#d61c6c','#b70b50','#b20a4d']
blues = ["#f7fbff", "#d2e3f3", "#b3d2e9", "#9ecae1", "#3082be", "#2171b5",  "#08306b"]
teals = ['#9cd1d2','#80c3c2','#66b3b2','#4da5a4','#349695','#258586','#157576']
tealblues = ['#9dd3d1','#81c3cb','#65b3c2','#45a2b9','#368fae','#347da0','#306a93']
G10 = px.colors.qualitative.G10


YEARS = [1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, \
		2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

BINS = ['0-2', '2.1-4', '4.1-6', '6.1-8', '8.1-10', '10.1-12', '12.1-14', \
		'14.1-16', '16.1-18', '18.1-20', '20.1-22', '22.1-24',  '24.1-26', \
		'26.1-28', '28.1-30', '>30']

DEFAULT_COLORSCALE = ["#2a4858", "#265465", "#1e6172", "#106e7c", "#007b84", \
	"#00898a", "#00968e", "#19a390", "#31b08f", "#4abd8c", "#64c988", \
	"#80d482", "#9cdf7c", "#bae976", "#d9f271", "#fafa6e"]
DEFAULT_OPACITY = 0.8

results = ['Potential Cause']


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#1B4F78',
    'opacity':'0.5',
    'color': '#B2EBF2',
}

app.layout = html.Div([
    html.H1(children='Wildfire Risk Prediction Dashboard',
        style={
            'textAlign': 'center',
            'color': '#306a93',
            'margin-bottom':'50px',
            'margin-top':'30px'
        }
    ),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Wildfire Map', value='tab-1-example-graph',children=[
            html.Div([
                html.P('Select a year on the slider below', 
                style={'text-align':'center', 'margin-top':40,'color':'#306a93'}
                ),
            ]),  
            html.Div([
                dcc.Slider(
                    id='years-slider',
                    min=1992,
                    max=2021,
                    step=1,
                    value=2021,
                    marks={str(year): 'All'  if year==2021 else str(year) for year in YEARS}
                    
                        ),
                    ], style={'width':1000, 'margin-left':200, 'margin-top':20,'color':'#306a93'}),   
        ],style={'background-color': '#1B4F78','opacity':'0.7'},selected_style=tab_selected_style),
        
        dcc.Tab(label='Number of Fire by States', value='tab-2-example-graph',children=[],style={'background-color': '#1B4F78','opacity':'0.7'},selected_style=tab_selected_style),
        
        dcc.Tab(label='Cause Breakdown', value='tab-3-example-graph',children=[],style={'background-color': '#1B4F78','opacity':'0.7'},selected_style=tab_selected_style),
        dcc.Tab(label='Prediction', value='tab-4-example-graph',children=[
            html.Div([
                html.Div([
                    html.Div([
            html.Div([
            html.Div([
                dbc.Card([
                    dbc.FormGroup([
                        dbc.Label("Latitude", size="md"),
                        dbc.Input(
                            id ='lat',
                            placeholder='Enter Latitude...',
                            type='number',
                            value=40.967200,
                            style={"margin":"15px"}
                        ),
                    ]),
                   dbc.FormGroup([
                        dbc.Label("Longitude", size="md"),
                        dbc.Input(
                        id = 'lon',
                        placeholder='Enter Longitude...',
                        type='number',
                        value=-117.784200,
                        style={"margin":"15px"}
                    ),
                   ]),
                   dbc.FormGroup([
                        dbc.Label("Fire Size", size="md"),
                        dcc.Input(
                        id = 'size',
                        type='number',
                        value=15.3,
                        style={"margin":"15px"}
                    ),
                   ]),
                   dbc.FormGroup([
                    dbc.Label("Fire Year", size="md"),
                    dcc.Dropdown(
                    options=[
                             {'label': '1992', 'value': 1992},
                             {'label': '1993', 'value': 1993},
                             {'label': '1994', 'value': 1994},
                             {'label': '1995', 'value': 1995},
                             {'label': '1996', 'value': 1996},
                             {'label': '1997', 'value': 1997},
                             {'label': '1998', 'value': 1998},
                             {'label': '1999', 'value': 1999},
                             {'label': '2000', 'value': 2000},
                             {'label': '2001', 'value': 2001},
                             {'label': '2002', 'value': 2002},
                             {'label': '2003', 'value': 2003},
                             {'label': '2004', 'value': 2004},
                             {'label': '2005', 'value': 2005},
                             {'label': '2006', 'value': 2006},
                             {'label': '2007', 'value': 2007},
                             {'label': '2008', 'value': 2008},
                             {'label': '2009', 'value': 2009},
                             {'label': '2010', 'value': 2010},
                             {'label': '2011', 'value': 2011},
                             {'label': '2012', 'value': 2012},
                             {'label': '2013', 'value': 2013},
                             {'label': '2014', 'value': 2014},
                             {'label': '2015', 'value': 2015},
                             {'label': '2016', 'value': 2016},
                             {'label': '2017', 'value': 2017},
                             {'label': '2018', 'value': 2018},
                             {'label': '2019', 'value': 2019},
                             {'label': '2020', 'value': 2020},
                             {'label': '2021', 'value': 2021},
                             ],
                    value='1997',
                    id='year-selected',
                    style={'width': '60%', 'margin': 15}
                ),

                   ]),
                   dbc.FormGroup([
                    dbc.Label("Fire Month", size="md"),
                    dcc.Dropdown(
                    options=[
                             {'label': 'Jan', 'value': 1},
                             {'label': 'Feb', 'value': 2},
                             {'label': 'Mar', 'value': 3},
                             {'label': 'Apr', 'value': 4},
                             {'label': 'May', 'value': 5},
                             {'label': 'Jun', 'value': 6},
                             {'label': 'Jul', 'value': 7},
                             {'label': 'Aug', 'value': 8},
                             {'label': 'Sep', 'value': 9},
                             {'label': 'Oct', 'value': 10},
                             {'label': 'Nov', 'value': 11},
                             {'label': 'Dec', 'value': 12},
                             ],
                    value=1,
                    id='month-selected',
                    style={'width': '60%', 'margin': 15}
                ),

                   ]),
                   dbc.FormGroup([
                    dbc.Label("State", size="md"),
                    dcc.Dropdown(
                    options=[
                             {'label': 'AL', 'value': 'AL'},
                             {'label': 'AK', 'value': 'AK'},
                             {'label': 'AZ', 'value': 'AZ'},
                             {'label': 'AR', 'value': 'AR'},
                             {'label': 'AS', 'value': 'AS'},
                             {'label': 'CA', 'value': 'CA'},
                             {'label': 'CO', 'value': 'CO'},
                             {'label': 'CT', 'value': 'CT'},
                             {'label': 'DE', 'value': 'DE'},
                             {'label': 'FL', 'value': 'FL'},
                             {'label': 'GA', 'value': 'GA'},
                             {'label': 'GU', 'value': 'GU'},
                             {'label': 'HI', 'value': 'HI'},
                             {'label': 'ID', 'value': 'ID'},
                             {'label': 'IL', 'value': 'IL'},
                             ],
                    value='CA',
                    id='state-selected',
                    style={'width': '60%', 'margin': 15}
                ),
                   ]),
                   dbc.FormGroup([
                    dbc.Label("Fire Size Class", size="md"),
                    dcc.Dropdown(
                    options=[
                             {'label': 'A', 'value': 'A'},
                             {'label': 'B', 'value': 'B'},
                             {'label': 'C', 'value': 'C'},
                             {'label': 'D', 'value': 'D'},
                             {'label': 'E', 'value': 'E'},
                             {'label': 'F', 'value': 'F'},
                             {'label': 'G', 'value': 'G'},
                             ],
                    value='C',
                    id='class-selected',
                    style={'width': '60%', 'margin': 15}
                ),
                   ]),
                   dbc.FormGroup([
                    dbc.Label("Model", size="md"),
                    dcc.Dropdown(
                    options=[
                             {'label': 'Random Forest', 'value': 'RF'},
                             {'label': 'Gradient Boosting Trees', 'value': 'GBT'},
                             {'label': 'Logistic Regression', 'value': 'LOG'},
                             {'label': 'Support Vector Machine', 'value': 'SVM'},
                            ],
                    value='RF',
                    id='model-selected',
                    style={'width': '60%', 'margin': 15}
                ),
                   ]),
                   dbc.FormGroup([
                    dbc.Label("Discovery Date", size="md"),
                    dcc.DatePickerSingle(
                    id='date-picker-single',
                    date=dt(1997, 5, 10),
                    style={'margin': 30}
                ),
                   ]),
                   dbc.FormGroup([
                   ]),
                   dbc.FormGroup([
                   ]), 
                ])               
                ], style={'margin-left': 30}),
                html.Button('Submit', id='button', style={"margin-left":"30px"})
            ], className='six columns',style={
                    'margin-left': '50px',
    }),

                    

                ], style={"flex":"50%"}),
                html.Div([
                         html.Div([
                        dcc.Graph(
                            id = 'result_chart',
                                    className='six columns')
            ], style={
                    'width': '90%',
                    'fontFamily': 'Sans-Serif',
                    'margin-left': 'auto',
                    'margin-top': '100px',
                    'margin-right': 'auto',
                    'color': '#f9f9f4'
    })

                ], style={"flex":"50%"}),
            ], style={"display":"flex"}),
            

       
        ]),
        ],
        
        style={'background-color': '#1B4F78','opacity':'0.7'},selected_style=tab_selected_style),
        dcc.Tab(label='Twitter Analysis', value='tab-5-example-graph',children=[
            html.Div([
                html.Iframe(src="https://datastudio.google.com/embed/reporting/0ef26deb-3fae-4938-aea8-0fd35d49d0ae/page/KuMAD",
                style={"height": "800px", "width": "80%", "border":0, "frameborder":0})
            ],style={"margin-left":200}),     
        ],style={'background-color': '#1B4F78','opacity':'0.7'},selected_style=tab_selected_style),
    ],style={
        'width': '100%',
        'fontFamily': 'Sans-Serif',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'color': '#f9f9f4'
    }),
    html.Div(
        id='tabs-content-example-graph',
        style={'display': 'flex','justify-content':'center','margin':'50px'},className = "six columns"
    )
])

@app.callback(
            #  Output('tabs-content-example-graph', 'figure'),
            Output('tabs-content-example-graph', 'children'),
            [Input('tabs-example-graph', 'value'),Input("years-slider", "value")])
def render_content(tab,year):
    if tab == 'tab-1-example-graph':
        if year==2021:
            data = []
            layout = dict(
                title = 'US Wildfires Map',
                # showlegend = False,
                autosize = False,
                width = 1300,
                height = 900,
                hovermode = False,
                legend = dict(
                    x=0.7,
                    y=-0.1,
                    bgcolor="rgba(255, 255, 255, 0)",
                    font = dict( size=11 ),
                )
            )
            years = sorted(df['FIRE_YEAR'].unique())
            for i in range(len(years)):
                geo_key = 'geo'+str(i+1) if i != 0 else 'geo'
                lons = list(df[ df['FIRE_YEAR'] == years[i] ]['LONGITUDE'])
                lats = list(df[ df['FIRE_YEAR'] == years[i] ]['LATITUDE'])
                data.append(
                    dict(
                        type = 'scattergeo',
                        showlegend=False,
                        lon = lons,
                        lat = lats,
                        geo = geo_key,
                        name = str(years[i]),
                        marker = dict(
                            color = "rgb(255, 90, 90)",
                            opacity = 0.5
                        )
                    )
                )

                data.append(
                    dict(
                        type = 'scattergeo',
                        showlegend = False,
                        lon = [-78],
                        lat = [47],
                        geo = geo_key,
                        text = [years[i]],
                        mode = 'text',
                    )
                )
                layout[geo_key] = dict(
                    scope = 'usa',
                    showland = True,
                    landcolor = 'rgb(180, 200, 229)',
                    showcountries = False,
                    domain = dict( x = [], y = [] ),
                    subunitcolor = "rgb(255, 255, 255)",
                )
            z = 0
            COLS = 5
            ROWS = 6
            for y in reversed(range(ROWS)):
                for x in range(COLS):
                    geo_key = 'geo'+str(z+1) if z != 0 else 'geo'
                    layout[geo_key]['domain']['x'] = [float(x)/float(COLS), float(x+1)/float(COLS)]
                    layout[geo_key]['domain']['y'] = [float(y)/float(ROWS), float(y+1)/float(ROWS)]
                    z=z+1
                    if z > 28:
                        break

            return html.Div([
                html.H3(dcc.Graph(
                    id='t1',
                    figure={"data": data, "layout": layout}))])
        else:
            data = []
            layout = dict(
                title = 'US Wildfires Map',
                # showlegend = False,
                autosize = True,
                # width = 700,
                # height = 700,
                hovermode = False,
                legend = dict(
                    x=0.7,
                    y=-0.1,
                    bgcolor="rgba(255, 255, 255, 0)",
                    font = dict( size=11 ),
                ) ,
                
            )
            
            geo_key = 'geo'
            lons = list(df[ df['FIRE_YEAR'] == year ]['LONGITUDE'])
            lats = list(df[ df['FIRE_YEAR'] == year ]['LATITUDE'])
            data.append(
                dict(
                    type = 'scattergeo',
                    showlegend=False,
                    lon = lons,
                    lat = lats,
                    geo = geo_key,
                    name = str(year),
                    marker = dict(
                        color = "rgb(255, 90, 90)",
                        opacity = 0.8
                    )
                )
            )

            data.append(
                dict(
                    type = 'scattergeo',
                    showlegend = False,
                    lon = [-78],
                    lat = [47],
                    geo = geo_key,
                    text = [year],
                    mode = 'text',
                )
            )
            layout[geo_key] = dict(
                scope = 'usa',
                showland = True,
                landcolor = 'rgb(180, 200, 229)',
                showcountries = False,
                domain = dict( x = [], y = [] ),
                subunitcolor = "rgb(255, 255, 255)",
            ) 
            layout[geo_key]['domain']['x'] = [0, 1]
            layout[geo_key]['domain']['y'] = [0, 1]         
            return html.Div([
                html.H3(dcc.Graph(
                    id='t1',
                    figure={"data": data, "layout": layout}))])

    elif tab == 'tab-2-example-graph':
        barscene = px.bar(
        df_states,
        width=900, 
        height=700,
        x=df_states["Total_Number_of_Fires"],
        y=df_states["STATE"].unique(),
        labels={"x": "Total_Number_of_Fires", "y": "States"},
        color=df_states["Total_Number_of_Fires"],
        color_continuous_scale=px.colors.sequential.Sunset,
        title="Total Number of Fires by States",
        orientation="h",
        )
        barscene.update_layout(title=dict(x=0.5), paper_bgcolor="rgb(203,213,232)",)
        return html.Div([
                html.H3(dcc.Graph(
                    id='t2',
                    figure=barscene))])
        
    elif tab == 'tab-3-example-graph':
        scatter = px.scatter(
        df_cause,
        width=900, 
        height=700,
        x="Year",
        y="Cause",
        size="Total_Fire_Size",
        color="Total_Fire_Size",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Wildfire Cause Breakdown",
        ).for_each_trace(lambda t: t.update(name=''))
        
        scatter.update_layout(
            xaxis_tickangle=30,
            title=dict(x=0.5),
            xaxis_tickfont=dict(size=9),
            yaxis_tickfont=dict(size=9),
            margin=dict(l=100, r=100, t=50, b=20),
            paper_bgcolor="rgb(203,213,232)",
        )
        return html.Div([
                html.H3(dcc.Graph(
                    id='t3',
                    figure=scatter))])

    elif tab == 'tab-4-example-graph':
        return html.Div([])

    elif tab == 'tab-5-example-graph':
        return html.Div([])

@app.callback(Output("result_chart", "figure"),
              [Input('button', 'n_clicks')],
              [State("lat", "value"),
               State("lon", "value"),
            #    State("drought-slider", "value"),
               State("model-selected", "value"),
               State("date-picker-single", "date"),
               State("year-selected", "value"),
               State("size", "value"),
               State("month-selected", "value"),
               State("state-selected", "value"),
               State("class-selected", "value"),
               ])

def update_figure(n_clicks, lat, lon, model, discoveryDate, fireYear, size, fireMonth, state,sizeClass):

    discoveryYear = int(discoveryDate.split(' ')[0][:4])
    discoveryMonth = int(discoveryDate.split(' ')[0][5:7])
    discoveryDay = int(discoveryDate.split(' ')[0][8:10])
    cr_date = datetime(discoveryYear, discoveryMonth, discoveryDay)
    fmt = '%y%j'
    dt = cr_date.strftime(fmt)
    # print(dt)
    # print(discoveryDate)
    # print(fireYear)
    # print(discoveryYear)
    # print(discoveryMonth)
    # print(discoveryDay)
    # drought_input = float(drought * 20)
    # lat = 40.967200
    # lon = 50.784200
    # x = [[fireYear, lat, lon, 14, '07033', 0.2, 1,6, discoveryMonth,6]]
  
    
    
    state_encoded = state_encoder.fit_transform([state])
    week_encoded = week_encoder.fit([discoveryDate])
    size_encoded = size_encoder.fit_transform([sizeClass])
    # print(week_encoded)
    x = pd.DataFrame(np.array([fireYear, lat, lon, 1, dt, size, 1,6, fireMonth]).reshape(1,-1),
    columns=['FIRE_YEAR', 'LATITUDE', 'LONGITUDE', 'STATE', 'DISCOVERY_DATE', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'DAY_OF_WEEK', 'MONTH'])
    X_test_scaled = scaler.transform(x)

    if model == 'RF':
        y_pred = rf_joblib.predict(X_test_scaled)
    elif model == 'GBT':
        y_pred = gbt_joblib.predict(X_test_scaled)
    elif model == 'LOG':
        y_pred = logistic_joblib.predict(X_test_scaled)
    elif model == 'SVM':
        y_pred = svm_joblib.predict(X_test_scaled)

    # cause_test = [[lat, lon, month, year, drought_input, y_pred[0]]]
    # cause_pred = rf_cause.predict(cause_test)

    # params = {'lat': lat,
    #           'lon': lon}
    # response_area = requests.get('https://geo.fcc.gov/api/census/area', params=params)
    # county_fips = response_area.json()['results'][0]['county_fips']
    if y_pred ==0:
        cause = "natural"
    elif y_pred ==1:
        cause= "Accidental"
    elif y_pred ==1:
        cause= "Malicious"
    elif y_pred ==1:
        cause= "Other Causes"
    y_pred[0] += 1

    trace = go.Bar(x=results, y=[y_pred[0]])
    fig=go.Figure(data=trace, layout_yaxis_range=[1,4],layout=go.Layout(title={'text': 'Prediction', 'yanchor': 'top', 'x': 0.5, 'y': 0.9},
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                                yaxis={'tickvals': [1, 2, 3,4]}
                                ))
    
    fig.add_annotation(dict(font=dict(color='blue',size=15),
                                        x=0,
                                        y=-0.19,
                                        showarrow=False,
                                        text="1:Natural, 2:Accidental, 3:Malicious, 4:Other Causes",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    return fig
    # {"data": [trace],
    #         "layout": go.Layout(title={'text': 'Prediction', 'yanchor': 'top', 'x': 0.5, 'y': 1},
    #                             paper_bgcolor='white',
    #                             plot_bgcolor='white',
    #                             yaxis={'tickvals': [1, 2, 3]}
    #                             )
    # }

if __name__ == '__main__':
    app.run_server(debug=True)

