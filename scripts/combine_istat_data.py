import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("DCIS_POPRES1_27072020130534278.csv")

    df = data.loc[(data.SEXISTAT1 == 9) &
                  (data.STATCIV2 == 99) &
                  (data.ITTER107.str.len() >= 5) &
                  (data.ETA1 != "TOTAL")][["Territorio", "ETA1", "Value"]]
    
    df.ETA1 = df.ETA1.str.replace("Y","")
    df.ETA1 = df.ETA1.str.replace("_GE","")
    df.Territorio = df.Territorio.str.replace("Valle d'Aosta / Vallée d'Aoste",
                                              "Valle d'Aosta")
    df.Territorio = df.Territorio.str.replace("Bolzano / Bozen", "Bolzano")
    df = df.astype({"ETA1": int})

    df_final = pd.DataFrame()

    # Three groups: 0-25, 25-65, 65+
    for provincia in df.Territorio.unique():
        df1 = df.loc[df["Territorio"] == provincia]

        a = df1[df1["ETA1"] <= 25].groupby(["Territorio"]).agg({"Value" : sum})
        b = df1[(df1["ETA1"] > 25) & (df1["ETA1"] <= 65)].groupby(["Territorio"]).agg({"Value" : sum})
        c = df1[df1["ETA1"] > 65].groupby(["Territorio"]).agg({"Value" : sum})

        tmp = pd.concat([a, b, c])
        tmp.reset_index(level=0, inplace=True)
        tmp = tmp.append(pd.DataFrame([
            {"Territorio": provincia,
             "Value": tmp.Value.sum()}]),
            ignore_index=True)
        tmp["Eta"] = ["0-25", "25-65", "65-100", "Total"]
        tmp["Percentage"] = tmp.Value.apply(lambda x: x / tmp.Value.values[-1])

        df_final = pd.concat([df_final, tmp])

    df_final.to_csv("pop_prov_age_3_groups_2020.csv", index=False)