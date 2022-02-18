import streamlit as st
from multiapp import MultiApp
from apps import home, showData, uploadData # import your app modules here

app = MultiApp()

st.title('Hackaton III ')
st.markdown("""Navigation
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Upload patient folder", uploadData.app)
app.add_app("Show Data and analysis", showData.app)
# The main app
app.run()
