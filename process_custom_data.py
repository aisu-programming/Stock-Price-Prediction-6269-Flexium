''' Libraries '''
import pandas as pd


''' Functions '''
def __calculate_addition_info(df):

    # Absolute values
    df['Length']    = df['High']  - df['Low']                          # 5
    df['BarLength'] = df['Close'] - df['Open']                         # 6
    df['RodLength'] = df['High']  - df[['Open', 'Close']].max(axis=1)  # 7
    df['PinLength'] = df['Low']   - df[['Open', 'Close']].min(axis=1)  # 8

    # Relative values
    df['OpenRatio']  =  0                                              # 9
    df['HighRatio']  =  df['High']  / df['Open']                       # 10
    df['LowRatio']   =  df['Low']   / df['Open']                       # 11
    df['CloseRatio'] =  df['Close'] / df['Open']                       # 12
    df['RiseFall']   = (df['Close'] - df['Open']) / df['Open']         # 13
    df['BarRatio']   =  0                                              # 14
    df['RodRatio']   =  0                                              # 15
    df['PinRatio']   =  0                                              # 16

    # Index
    df['RSI_5']  = 0                                                   # 17
    df['RSI_10'] = 0                                                   # 18
    # df['K_9']    = 0                                                   # 19
    # df['D_9']    = 0                                                   # 20

    df_values = df.values
    for i in range(len(df_values)):

        if i == 0: continue
        df_values[i][9]  = df_values[i][1] / df_values[i-1][4]    # OpenRatio

        if df_values[i][5] != 0:
            df_values[i][14] = df_values[i][6] / df_values[i][5]  # BarRatio
            df_values[i][15] = df_values[i][7] / df_values[i][5]  # RodRatio
            df_values[i][16] = df_values[i][8] / df_values[i][5]  # PinRatio

        # RSI_5
        if i < 5: continue
        rise, fall = 0, 0
        for j in range(5):
            if df_values[i-j][13] > 0: rise += abs(df_values[i-j][13])
            else                     : fall += abs(df_values[i-j][13])
        rise /= 5
        fall /= 5
        df_values[i][17] = rise / (rise + fall) * 100

        # RSI_10
        if i < 10: continue
        rise, fall = 0, 0
        for j in range(10):
            if df_values[i-j][13] > 0: rise += abs(df_values[i-j][13])
            else                     : fall += abs(df_values[i-j][13])
        rise /= 10
        fall /= 10
        df_values[i][18] = rise / (rise + fall) * 100

    df_values = df_values[1:]
    
    return pd.DataFrame(df_values, columns=df.columns)


def combine_and_save(additional_info=False):

    stock_index  = pd.read_csv(r"data/custom/stock_index.csv")
    adapted_6269 = pd.read_csv(r"data/custom/6269_adapt.csv")
    
    if additional_info:
        stock_index  = __calculate_addition_info(stock_index)
        adapted_6269 = __calculate_addition_info(adapted_6269)
    adapted_6269 = adapted_6269.drop(columns=['Date'])
    stock_index  = stock_index.rename(columns={ c: f'SI_{c}' for c in stock_index.columns })
    stock_index  = stock_index.rename(columns={ 'SI_Date': 'Date' })

    combined_data = pd.concat([stock_index, adapted_6269], axis=1)
    combined_data_filename = "combined_addinfo" if additional_info else "combined"
    combined_data.to_csv(f"data/custom/{combined_data_filename}.csv", index=False)
    return


def main():
    # combine_and_save(additional_info=False)
    combine_and_save(additional_info=True)
    return


''' Execution '''
if __name__ == '__main__':
    main()