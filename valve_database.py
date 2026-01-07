# valve_database.py
import os
import pandas as pd
from valve import Valve

VALVE_DATA_FILE = "valve_database.xlsx"

def get_valve_sheet_name(valve):
    """Generate standardized sheet name for a valve"""
    return f"Valve_{valve.size}_{valve.rating_class}_{valve.valve_type}"

def load_valves_from_excel():
    """Load valve data from Excel file with note column"""
    try:
        if not os.path.exists('valve_database.xlsx'):
            return []
        
        df = pd.read_excel('valve_database.xlsx')
        valves = []
        
        for _, row in df.iterrows():
            # Parse Cv table
            cv_table = {}
            for col in [col for col in df.columns if col.startswith('Cv_')]:
                opening = int(col.split('_')[1])
                cv_table[opening] = row[col]
            
            # Parse Fl table
            fl_table = {}
            for col in [col for col in df.columns if col.startswith('Fl_')]:
                opening = int(col.split('_')[1])
                fl_table[opening] = row[col]
            
            # Parse Xt table
            xt_table = {}
            for col in [col for col in df.columns if col.startswith('Xt_')]:
                opening = int(col.split('_')[1])
                xt_table[opening] = row[col]
            
            # Get note column (default to empty string if not present)
            note = row.get('Note', '')
            
            valve = Valve(
                size_inch=row['Size_inch'],
                rating_class=row['Rating_Class'],
                cv_table=cv_table,
                fl_table=fl_table,
                xt_table=xt_table,
                fd=row.get('Fd', 1.0),
                d_inch=row.get('Diameter_inch', row['Size_inch']),
                valve_type=row.get('Valve_Type', 3),
                note=note
            )
            valves.append(valve)
        
        return valves
    except Exception as e:
        print(f"Error loading valve database: {e}")
        return []

def save_valves_to_excel(valves):
    """Save list of Valve objects to Excel with new structure"""
    # Create main valve list
    main_data = []
    valve_sheets = {}
    
    for valve in valves:
        sheet_name = get_valve_sheet_name(valve)
        main_data.append({
            'size': valve.size,
            'rating_class': valve.rating_class,
            'valve_type': valve.valve_type,
            'fd': valve.fd,
            'diameter': valve.diameter,
            'sheet_name': sheet_name
        })
        
        # Create valve data table
        all_openings = sorted(set(list(valve.cv_table.keys()) + 
                              list(valve.fl_table.keys()) + 
                              list(valve.xt_table.keys())))
        
        valve_data = []
        for opening in all_openings:
            valve_data.append({
                'Opening (%)': opening,
                'Cv': valve.cv_table.get(opening, None),
                'Fl': valve.fl_table.get(opening, None),
                'Xt': valve.xt_table.get(opening, None)
            })
        
        valve_sheets[sheet_name] = pd.DataFrame(valve_data)
    
    # Create Excel writer
    with pd.ExcelWriter(VALVE_DATA_FILE, engine='openpyxl') as writer:
        # Save main list
        pd.DataFrame(main_data).to_excel(
            writer, sheet_name='Valve List', index=False
        )
        
        # Save each valve sheet
        for sheet_name, df in valve_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def add_valve_to_database(valve):
    """Add valve to Excel database with note"""
    try:
        # Check if file exists
        if os.path.exists('valve_database.xlsx'):
            df = pd.read_excel('valve_database.xlsx')
        else:
            # Create new DataFrame
            df = pd.DataFrame()
        
        # Prepare valve data
        valve_data = {
            'Size_inch': valve.size,
            'Rating_Class': valve.rating_class,
            'Valve_Type': valve.valve_type,
            'Fd': valve.fd,
            'Diameter_inch': valve.diameter,
            'Note': valve.note  # Add note
        }
        
        # Add Cv values
        for opening, cv in valve.cv_table.items():
            valve_data[f'Cv_{opening}'] = cv
        
        # Add Fl values
        for opening, fl in valve.fl_table.items():
            valve_data[f'Fl_{opening}'] = fl
        
        # Add Xt values
        for opening, xt in valve.xt_table.items():
            valve_data[f'Xt_{opening}'] = xt
        
        # Add to DataFrame
        new_row = pd.DataFrame([valve_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to Excel
        df.to_excel('valve_database.xlsx', index=False)
        return True
    except Exception as e:
        print(f"Error adding valve to database: {e}")
        return False

def delete_valve_from_database(size, rating_class, valve_type):
    """Delete valve from Excel database"""
    try:
        if not os.path.exists('valve_database.xlsx'):
            return False
        
        df = pd.read_excel('valve_database.xlsx')
        
        # Filter out the valve to delete
        mask = ~((df['Size_inch'] == size) & 
                 (df['Rating_Class'] == rating_class) & 
                 (df['Valve_Type'] == valve_type))
        df = df[mask]
        
        df.to_excel('valve_database.xlsx', index=False)
        return True
    except Exception as e:
        print(f"Error deleting valve from database: {e}")
        return False
