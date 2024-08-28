import os
import pandas as pd
import peptides
import argparse

def CDRH3_Comb(file_path, start_r, end_r):
    file_name = f"{file_path}_Paresd_Result.csv"
    start_r -= 1
    
    with open(file_path, "r") as file:
        data = file.read()

    ref_line = data.strip().split("\n")[1]
    lines = data.strip().split("\n")[2:]
    
    data_list = []
    for i in range(0, len(lines), 2):
        info_line = lines[i]
        seq_line = lines[i+1]
        
        info_parts = info_line.split(",")
        info_dict = {part.split("=")[0][1:]: part.split("=")[1] for part in info_parts}
        info_dict["sequence"] = seq_line
        data_list.append(info_dict)
    import pandas as pd
    df = pd.DataFrame(data_list)
    sequence_slice = df['sequence'].str[start_r:end_r]
    reference_sequence = ref_line[start_r:end_r]
    print("Original sequence is: " + reference_sequence)
    df1 = pd.DataFrame({
        'combination': sequence_slice,
        'seq_recovery': df['seq_recovery']
    })
    
    df2 = df1.groupby(['combination', 'seq_recovery']).size().reset_index(name='count')
    total_count = df2['count'].sum()
    df2['percentage'] = ((df2['count'] / total_count) * 100).round(2)
    df2['percentage'] = df2['percentage'].astype(str) + '%'
    df2 = df2.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    new_row = pd.DataFrame([[reference_sequence, 1, 0, 0]], columns=df2.columns)
    df2 = pd.concat([new_row, df2], ignore_index=True)
    
    for index, row in df2.iterrows():
        seq=row["combination"]
        peptide = peptides.Peptide(seq)
        
        descriptor_names = ['volume', 'hydro']
        for i, descriptor_value in enumerate(peptide.physical_descriptors()):
            column_name = f'PD{i+1}_{descriptor_names[i]}'
            df2.at[index, column_name] = f"{descriptor_value:.4f}"
        
        df2.at[index, 'PD3_charge'] =  f"{peptide.charge(pH=7.4):.4f}"
    
    target_sequence = reference_sequence
    changes = []
    
    for index, row in df2.iterrows():
        combination = row['combination']
        change = [(target_sequence[i], i+start_r+1, combination[i]) for i in range(len(target_sequence)) if combination[i] != target_sequence[i]]
        changes.append(change)
    
    df2['changes'] = changes
    
    df3 = pd.DataFrame()
    
    for index, row in df2.iterrows():
        changes = row['changes']
        for change in changes:
            original_aa, position, new_aa = change
            if position not in df3.columns:
                df3[position] = ''
            df3.at[index, position] = new_aa
    
    df3.fillna('/', inplace=True)
    df3 = df3.reindex(sorted(df3.columns), axis=1)
    
    dfn = pd.DataFrame()
    dfn['num_changes'] = df2['changes'].apply(len)
    df4 = pd.concat([df2, dfn[['num_changes']], df3], axis=1)
    
    df4.to_csv(file_name, index=False)

    print(file_path + " has been parsed!")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir' , help='Directory to look for .fa files')
    parser.add_argument('--start', type=int, help='Start position')
    parser.add_argument('--end', type=int, help='End position')

    args = parser.parse_args()
    
    files = [f for f in os.listdir(args.dir) if f.endswith('.fa')]
    for file in files:
        file_path = os.path.join(args.dir, file) 
        CDRH3_Comb(file_path, args.start, args.end)

if __name__ == "__main__":
    main()