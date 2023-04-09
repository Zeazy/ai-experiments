CUSTOM_CSS = """
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .centered-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
        }
        
        .prediction {
            font-size: 32px;
            font-weight: bold;
            color: #5C5C5C;
        }

        .probabilities {
            font-size: 18px;
            font-weight: 600;
            color: #5C5C5C;
        }
        
        /* Add button hover effect */
        .button:hover {
            background-color: red;
        }

        /* Style Streamlit table */
        .dataframe {
            font-size: 14px;
            color: black;
            background-color: white;
            border: 1px solid #d3d3d3;
        }
        .dataframe th {
            font-weight: bold;
            background-color: #e6e6e6;
            border: 1px solid #d3d3d3;
        }
        .dataframe td {
            border: 1px solid #d3d3d3;
        }
    """