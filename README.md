# hypoxia-lab-openox-dashboard

# What I've changed:
- ‘Monk ABC’ -> ‘Monk ABCD’
- ‘Monk DEF’ -> ‘Monk EFG’
- Removed gradient color and the progress bar
- Removed column related to BMI and Age
- Changed the height in st.dataframe (reference: https://discuss.streamlit.io/t/st-dataframe-controlling-the-height-threshold-for-scolling/31769/4 )
- Changed all the “Patients” to “Subjects”
- Added more ISO criteria checks

# Requests from Mike
- Freeze the first two columns so they don’t disappear when scrolling to the right
- Add explanation text to show when hovering over the column header. Header could be shorter
- Highlight the model/manufacture if every criteria is met
