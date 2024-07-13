
##################################
######### COLOR DEFINITIONS  #####
## and skin color groupings ######
##################################

#fitzpatrick scale color definitions
fpcolors = {'I - Pale white skin': '#f4d0b0',
        'II - White skin':'#e8b48f',
        'III - Light brown skin':'#d39e7c',
        'IV - Moderate brown skin':'#bb7750',
        'V - Dark brown skin':'#a55d2b',
        'VI - Deeply pigmented dark brown to black skin': '#3c201d'}
"""
Dictionary mapping Fitzpatrick skin types to their corresponding color codes.
Example:
- 'I - Pale white skin': '#f4d0b0'
"""

#fitzpatrick scale color definitions
fpcolors2 = {1: '#f4d0b0',
        2:'#e8b48f',
        3:'#d39e7c',
        4:'#bb7750',
        5:'#a55d2b',
        6: '#3c201d'}
"""
Dictionary mapping numeric Fitzpatrick skin types to their corresponding color codes.
Example:
- 1: '#f4d0b0' (corresponds to 'I - Pale white skin')
"""

#fitzpatrick scale color definitions - 
fpcolors3 = {'I':'#f4d0b0',
        'II':'#e8b48f',
        'III':'#d39e7c',
        'IV':'#bb7750',
        'V':'#a55d2b',
        'VI': '#3c201d'}
"""
Dictionary mapping abbreviated Fitzpatrick skin types to their corresponding color codes.
This dictionary is an updated version that uses abbreviations for types.
Example:
- 'I': '#f4d0b0'
"""

#monk scale color definitions
# mscolors= {'A': '#f6ede4',
#             'B': '#f3e7db',
#             'C': '#f7ead0',
#             'D': '#eadaba',
#             'E': '#d7bd96',
#             'F': '#a07e56',
#             'G': '#825c43',
#             'H': '#604134',
#             'I': '#3a312a',
#             'J': '#292420'}

# https://osf.io/preprints/socarxiv/pdf4c - this is the values used in the print out/Ellis shared
mscolors = {'A': '#f7ede4', 
            'B': '#f3e7da', 
            'C': '#f6ead0', 
            'D': '#ead9bb', 
            'E': '#d7bd96', 
            'F': '#9f7d54', 
            'G': '#815d44', 
            'H': '#604234', 
            'I': '#3a312a', 
            'J': '#2a2420'}
"""
Dictionary mapping Monk scale color grades to their corresponding color codes.
This version uses values shared in the referenced preprint.
Example:
- 'A': '#f7ede4'
"""

#define monk skin tone categories
mstlight = ['A','B','C']
mstmedium = ['D','E','F','G']
mstdark = ['H','I','J']

##################################