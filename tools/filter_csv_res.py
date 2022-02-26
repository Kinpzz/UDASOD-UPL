import os

src_csv_path = ""
out_csv_file = ""

orginal_metrics  = [
    'sm', 
    'wfm', 
    'adp_fm', 'mean_fm', 'max_fm', 
    'adp_em', 'mean_em', 'max_em', 
    'mae'
]

target_metrics = [
    'sm', 'wfm', 'mae'
]
target_metrics = [ 
    'sm','wfm','mean_fm','max_fm','mean_em','max_em','mae'
]

if __name__ == "__main__":
    
    with open(src_csv_path) as f:
        res = f.readlines()
        out_lines = []
        for line in res:
            res_dict = {}
            mets = line.split(',')[1:]

            c = 0
            while c < len(mets):
                ds_name = c
                res_dict[ds_name] = {}
                for met in orginal_metrics:
                    res_dict[ds_name][met] = met[c] 
                    c += 1
            out_line = [mets[0]]
            for k in res_dict: # orders matters
                print(k)
                for met in target_metrics:
                    out_line.append(res_dict[k][met])
            out_lines.append(','.join(out_line))
    
    print('\n'.join(out_lines))
    with open(out_csv_file, 'w') as f:
        f.writelines(out_lines)
        
            
        
            
