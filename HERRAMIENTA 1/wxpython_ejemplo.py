import wx, os
import pandas as pd
from wx.core import Icon
import matplotlib.pyplot as plt

class MyPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        ## Global Variables
        self.sortAsc = None

        self.my_text = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.my_text.SetFont(wx.Font(10, family = wx.DEFAULT, style = wx.NORMAL, weight = wx.BOLD, faceName = 'Consolas'))

        btn1 = wx.Button(self, label='Open Excel File')
        btn1.Bind(wx.EVT_BUTTON, self.onOpen)

        btn2 = wx.Button(self, label='Sort',size=(70, 30))
        btn2.Bind(wx.EVT_BUTTON, self.sorter)

        btn3 = wx.Button(self, label='Graph',size=(70, 30))
        btn3.Bind(wx.EVT_BUTTON, self.charts)

        sizer = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        sizer.Add(self.my_text, 1, wx.ALL|wx.EXPAND)
        sizer.Add(btn1, 0, wx.ALL|wx.CENTER, 5)

        hbox.Add(btn2, flag=wx.LEFT,border=5)
        hbox.Add(btn3, flag=wx.LEFT,border=5)

        sizer.Add(hbox, flag=wx.CENTER, border=5)
        sizer.Add((-1, 5))

        self.SetSizer(sizer)

    def onOpen(self, event):
        
        wildcard = "Excel files (*.xlsx)|*.xlsx"
        dialog = wx.FileDialog(self, "Open Excel Files", wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        
        path = dialog.GetPath()
        
        if os.path.exists(path):
            my_sheet = 'BAS_LIM_DISTRITOS' # change it to your sheet name
            cols={'ID','NOMBDIST','NOMBPROV','NOMBDEP',"SHAPE_AREA"}
            self.df = pd.read_excel(path, sheet_name = my_sheet, usecols=cols)
            #
            self.my_text.WriteText(self.df[:30].to_string(index=False,col_space=20,max_colwidth=15,justify='center')+'\n')#,col_space=30,max_colwidth=25
    
    def sorter(self, event):
        ##
        try:
            self.df.head()
        except:
            print("No se ha cargado el archivo Excel")
            return
        ##
        self.my_text.Clear()
        #
        if self.sortAsc != None:
            if self.sortAsc == True:
                self.sortAsc = False
            else:
                self.sortAsc = True
        else:
            self.sortAsc = True
        df = self.df.sort_values('SHAPE_AREA',ascending=self.sortAsc)
        self.my_text.WriteText(df[:30].to_string(index=False,col_space=20,max_colwidth=15,justify='center')+'\n')

    def charts(self, event):
        ##
        try:
            self.df.head()
        except:
            print("No se ha cargado el archivo Excel")
            return
        ##
        areas = []
        depas = []
        for depa in  self.df['NOMBDEP'].unique():
            temp=self.df.loc[self.df['NOMBDEP'] == depa]
            areas.append(sum(temp["SHAPE_AREA"]))
            depas.append(depa)

        df_dep = pd.DataFrame(list(zip(depas,areas)),columns =['DEPA', 'AREA'])
        df_dep = df_dep.sort_values('AREA')

        fig1, ax1 = plt.subplots()
        ax1.title.set_text('Gráfico Circular')
        fig1.canvas.set_window_title('Gráfico Circular')
        ax1.pie(df_dep['AREA'], labels=df_dep['DEPA'], autopct='%1.1f%%', startangle=90,textprops={'fontsize': 8})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

class MyFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Excel File Reader',size=(800,600))
        logo=wx.Icon("./img/logo_jpi.png")
        self.SetIcon(logo)

        panel = MyPanel(self)

        self.Show()

if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()
