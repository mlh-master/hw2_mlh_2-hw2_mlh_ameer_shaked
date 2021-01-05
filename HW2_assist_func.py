### assisting function for HW2 ###

def vis_table():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.model_selection import train_test_split as tts

    ## data preprocessing
    filepath = Path.cwd().joinpath('HW2_data.csv')
    T1D = pd.read_csv(filepath)
    T1D_raw_exnan = T1D.dropna()
    diagnosis = T1D_raw_exnan['Diagnosis']
    age = T1D_raw_exnan['Age']
    feat_col = set(T1D_raw_exnan.columns)
    feat_col.remove('Diagnosis')
    feat_col.remove('Age')
    T1D_exnan = T1D_raw_exnan[feat_col]
    # convert values to binary 0 or 1:
    # male = 0, female = 1
    T1D_exnan = T1D_exnan.replace('Yes', 1)
    T1D_exnan = T1D_exnan.replace('No', 0)
    T1D_exnan = T1D_exnan.replace('Female', 1)
    T1D_exnan = T1D_exnan.replace('Male', 0)

    xtr, xte, ytr, yte = tts(T1D_exnan, np.ravel(diagnosis), test_size=0.2, random_state=7, stratify=np.ravel(diagnosis))

    ## data visualization

    vis_table = pd.DataFrame(index=feat_col, columns=['Train [%]', 'Test [%]', 'Delta [%]'])
    for ii, col in enumerate(feat_col):
        vis_table.loc[col, 'Train [%]'] = 100 * xtr[col].sum() / xtr[col].count()
        vis_table.loc[col, 'Test [%]'] = 100 * xte[col].sum() / xte[col].count()
        vis_table.loc[col, 'Delta [%]'] = vis_table.loc[col, 'Train [%]'] - vis_table.loc[col, 'Test [%]']

    #xtr['Itching'].sum()
    #xtr.sum()

    #vis_table.index.name = 'Positive Feature'
    vis_table=vis_table.rename_axis('Positive Feature', axis=1)
    return vis_table



def feature_label():
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.model_selection import train_test_split as tts
    filepath = Path.cwd().joinpath('HW2_data.csv')
    T1D = pd.read_csv(filepath)
    T1D_raw_exnan = T1D.dropna()
    diagnosis = T1D_raw_exnan['Diagnosis']
    age = T1D_raw_exnan['Age']
    feat_col = set(T1D_raw_exnan.columns)
    feat_col.remove('Diagnosis')
    feat_col.remove('Age')
    T1D_exnan = T1D_raw_exnan[feat_col]
    # convert values to binary 0 or 1:
    # male = 0, female = 1
    T1D_exnan = T1D_exnan.replace('Yes', 1)
    T1D_exnan = T1D_exnan.replace('No', 0)
    T1D_exnan = T1D_exnan.replace('Male', 0)
    T1D_exnan = T1D_exnan.replace('Female', 1)
    diagnosis = diagnosis.replace('Positive', True).replace('Negative', False)
    pos_diag = T1D_exnan[diagnosis]
    neg_diag = T1D_exnan[~diagnosis]
    col = 'Gender'

    #sns.set_theme(style="ticks", color_codes=True)
    #fig, axs = plt.subplots(4, 4)
    #for ii, col in enumerate(feat_col):
     #   axs[ii] = sns.catplot(y=col, hue="Diagnosis", kind="count", data=T1D_raw_exnan)
    fig = plt.figure(figsize=(30,30))
    sns.set_theme(style="ticks", color_codes=True)
    #fig, axs = plt.subplots(16, 1)
    for ii, col in enumerate(feat_col):
        ax = fig.add_subplot(4,4,ii+1)
        plot = sns.countplot(y=col, hue="Diagnosis", data=T1D_raw_exnan, palette="rocket", edgecolor=".7",ax=ax)


    #sns.set_theme(style="ticks", color_codes=True)
    ##fig, axs = plt.subplots(16, 1)
    #for ii, col in enumerate(feat_col):
        #plot = sns.catplot(y=col, hue="Diagnosis", kind="count", data=T1D_raw_exnan, palette="magma", edgecolor=".6")