from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.animation as animation
from scipy import integrate
from scipy.interpolate import LSQUnivariateSpline
from copy import copy
from math import ceil, atan, sin, cos, sqrt, e

################################### funciones ###################################

def BaseLineCorrection(at, dt=0.01, type='polynomial', order=2, dspline=1000):
    """
    Realiza una corrección por Línea Base a un array de aceleraciones

    PARÁMETROS:
    at      : narray de aceleraciones 
    dt      : delta de tiempo en seguntos. para itk=0.01s
    type    : método de ajuste ('polynomial', 'spline')
    order   : orden del polinomio de aproximación para la línea base
    dspline : en caso de ser el método spline, define cada cuantos puntos se debe hacer el ajuste

    RETORNOS:
    at  : señal de aceleraciones corregida
    """
    # vt = integrate.cumtrapz(at, dx=dt, initial=0.0)
    x = np.arange(len(at))
    
    if type=='Polinomial':
        fit_at = np.polyval(np.polyfit(x, at, deg=order), x)
        
    if type =='Spline':
        splknots = np.arange(dspline / 2.0, len(at) - dspline / 2.0 + 2, dspline)
        spl = LSQUnivariateSpline(x=x, y=at, t=splknots, k=order)
        fit_at = spl(x)

    return at - fit_at

def GL(f, fl, n):
    """
    Hace un low cut al array de frecuencias

    inputs:
        f   : array de frecuencias
        fl  : low cut frecuency
        n   : orden de corte
    output:
        GH  : Ganancia de frecuencias recortada (array)
	    GL = ( (f/fl)**(2*n)/(1 + (f/fl)**(2*n)) )**0.5
    """
    return ( (f/fl)**(2*n)/(1 + (f/fl)**(2*n)))**0.5

def GH(f, fh, n):
    """
    Hace un high cut al array de frecuencias

    input:
        f   : array de frecuencias
        fh  : high cut frecuency
        n   : orden de corte
    output:
        GH  : Ganancia de frecuencias recortada (array)
	    GH = ( 1/(1 + (f/fh)**(2*n)) )**0.5
    """
    return ( 1/(1 + (f/fh)**(2*n)) )**0.5

def GB(f, fl, fh, n):
    """
    Hace un Butterworth al array de frecuencias

    inputs:
        f   : array de frecuencias
        fl  : low cut frecuency
        fh  : high cut frecuency
        n   : orden de corte
    output:
        GB  : Ganancia de frecuencias recortada (array)
    """
    return GL(f, fl, n)*GH(f, fh, n)

def Butterworth_Bandpass(signal, dt, fl, fh, n):
    """
    Hace un Butterworth Bandpass a las frecuencias de la señal

    inputs:                                         examples:
        signal      : señal (array)                         | array de aceleraciones
        dt          : delta de tiempo de la señal           | para itk = 0.01 seg
        fl          : low cut frecuency                     | fl = 0.10 Hz
        fh          : high cut frecuency                    | hf = 40.0 Hz
        n           : orden de corte                        | n = 15
    output:
        filter      : señal filtrada (array)
    """
    FFT = np.fft.rfft(signal)
    f = np.fft.rfftfreq(len(signal), d = dt)
    FFT_filtered = GL(f, fl, n)*FFT*GH(f, fh, n)

    return np.fft.irfft(FFT_filtered)

##################################################################################

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.curFile = ''
        self.setCurrentFile('')
        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()
        self.readSettings()

        self.viewStart()

    def open(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)

        fileName, filtr = QFileDialog.getOpenFileName(self, "Open File", "./", "(*.csv )")
        if fileName:
            self.loadFile(fileName)

        QApplication.restoreOverrideCursor()

    def loadFile(self, fileName):
        file = QFile(fileName)
        if not file.open(QFile.ReadOnly | QFile.Text):
            QMessageBox.warning(self, "Aplicacion",
                    "No se puedo leer el Archivo %s:\n%s." % (fileName, file.errorString()))
            return

        self.df = pd.read_csv(fileName, sep = ';', names = ["Time", "X", "Y", "Z"])
        n = self.df.shape[0]
        self.df.insert(0, 'N° Row', [i+1 for i in range(n)])

        inf = QTextStream(file)
        self.setCurrentFile(fileName)
        self.statusBar().showMessage("Archivo leido", 2000)

        self.baseLineAct.setEnabled(False)
        self.passBandAct.setEnabled(False)
        self.simuladAct.setEnabled(False)

        self.viewLoad()

    def about(self):
        QMessageBox.about(self, "Acerca de la aplicacion",
                "La <b>Aplicacion</b> es un ejemplo de como crear un MainWindow")

    def createActions(self):

        self.openAct = QAction(QIcon('./images/open.png'),
                "Abrir...", self, shortcut=QKeySequence.Open,
                statusTip="Abrir un archivo", triggered=self.open)

        self.exitAct = QAction(QIcon('./images/log-out.png'),
                "Salir", self, shortcut="Ctrl+Q",
                statusTip="Salir de la aplicacion", triggered=self.close)

        self.aboutAct = QAction("&Acerca de la aplicacion", self,
                statusTip="Muestra una descripcion acerca de la aplicacion",
                triggered=self.about)

        self.baseLineAct = QAction(QIcon('./images/baseLine.png'), "Linea Base",
                self, shortcut = 'Ctrl+L', statusTip = "Hace una correcion por Linea Base",
                triggered = self.viewBaseLine)

        self.passBandAct = QAction(QIcon('./images/passband.png'), "Pasa Banda",
                self, shortcut = 'Ctrl+L', statusTip = "Hace un Filtro por Pasa Banda ButterWorth",
                triggered = self.viewPassBand)

        self.simuladAct = QAction(QIcon('./images/simula.png'), "Simulacion",
                self, shortcut = 'Ctrl+L', statusTip = "Simulacion lineal de un MDOF",
                triggered = self.viewSimula)        

        self.baseLineAct.setEnabled(False)
        self.passBandAct.setEnabled(False)
        self.simuladAct.setEnabled(False)

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("Archivo")
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.correctMenu = self.menuBar().addMenu("Correciones")
        self.correctMenu.addAction(self.passBandAct)
        self.correctMenu.addAction(self.baseLineAct)

        self.simulaMenu = self.menuBar().addMenu("Simulacion")
        self.simulaMenu.addAction(self.simuladAct)

        self.helpMenu = self.menuBar().addMenu("Ayuda")
        self.helpMenu.addAction(self.aboutAct)

    def createToolBars(self):
        self.fileToolBar = self.addToolBar("Archivo")
        self.fileToolBar.addAction(self.openAct)
        self.fileToolBar.addAction(self.exitAct)

        self.correctToolBar = self.addToolBar("Correciones")
        self.correctToolBar.addAction(self.passBandAct)
        self.correctToolBar.addAction(self.baseLineAct)

        self.simulaToolBar = self.addToolBar("Simulacion")
        self.simulaToolBar.addAction(self.simuladAct)

    def createStatusBar(self):
        self.statusBar().showMessage("Listo")

    def readSettings(self):
        w_logo = QIcon('./images/logo.png')
        self.setWindowIcon(w_logo)
        settings = QSettings("Trolltech", "Application Example")

        ### Tamano de inicio ###
        rootWindow = QApplication.desktop() # retorna un objeto de la clase 'QDesktopWidget' la cual proporciona acceso a la información de la pantalla del sistema
        geometry = rootWindow.availableGeometry(self)
        W = int(geometry.width()*0.7)
        H = int(geometry.height()*0.8)
        self.resize(W, H)

        #### Mover centro ###
        geometryMainwindow = self.frameGeometry()
        centerPoint = geometry.center()
        geometryMainwindow.moveCenter(centerPoint)
        self.move(geometryMainwindow.topLeft())

    def setCurrentFile(self, fileName):
        self.curFile = fileName
        self.setWindowModified(False)

        if self.curFile:
            shownName = self.strippedName(self.curFile)
        else:
            shownName = '---'

        self.setWindowTitle("%s[*] | APLICACION DEMOSTRATIVA" % shownName)

    def strippedName(self, fullFileName):
        return QFileInfo(fullFileName).fileName()

##################################### Vistas #####################################
    def viewStart(self):
        self.centralwidget = QWidget(self)
        layout = QGridLayout(self.centralwidget)
        picture = QPixmap("./images/seismograph.png")
        labelPicture = QLabel()
        labelPicture.setAlignment(Qt.AlignCenter)
        labelPicture.setPixmap(picture)
        layout.addWidget(labelPicture)
        self.setCentralWidget(self.centralwidget)

    def viewLoad(self):

        def okButton():
            self.t = np.array(self.df['Time'])
            self.dt = self.t[1] - self.t[0]
            self.acc = [np.array(self.df['X']), np.array(self.df['Y']), np.array(self.df['Z'])]
            self.acc_corr = copy(self.acc)

            self.baseLineAct.setEnabled(True)
            self.passBandAct.setEnabled(True)
            self.simuladAct.setEnabled(True)

            self.centralwidget.deleteLater()
            self.viewStart()

        def cancelButton():
            self.centralwidget.deleteLater()
            self.viewStart()

        def genGraphs():

            sig = [np.array(self.df['X']), np.array(self.df['Y']), np.array(self.df['Z'])]
            t = np.array(self.df['Time'])
            max_lim = max([ np.max(np.abs(sig[i])) for i in range(3)] )
            w = 0.5
            colors = ['b', 'g', 'k']
            direct = ['X', 'Y', 'Z']

            fig, axs = plt.subplots(3)
            for i in range(3):
                axs[i].plot(t, sig[i], colors[i],  lw = w , label= 'pico: ' + str(round(np.max(np.abs(sig[i])), 2)) + ' cm/s^2')
                axs[i].xaxis.set_tick_params(labelsize=6)
                axs[i].yaxis.set_tick_params(labelsize=6)
                axs[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'small')
                axs[i].set_ylabel(ylabel='Aceleración en %s ($cm/s^2$)' %direct[i], fontsize= 'small')
                axs[i].legend(loc='upper right', frameon=True, fontsize= 'small', handlelength=2.0)
                axs[i].label_outer()
                axs[i].set_xlim(t[0], t[-1])
                axs[i].set_ylim(-max_lim*1.05 , max_lim*1.05)
                axs[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

            plt.subplots_adjust(left=0.085, bottom=0.085, right=0.98, top=0.97, wspace=0.2, hspace=0.15)

            self.figwindow = QMainWindow()
            canvasWidget = QWidget(self.figwindow)
            layout = QHBoxLayout(canvasWidget)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.figwindow.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(canvas, self))
            self.figwindow.setCentralWidget(canvasWidget)

        self.centralwidget = QWidget(self)
        self.cwHLyt = QHBoxLayout(self.centralwidget)

        self.gb_1_2_VLyt = QVBoxLayout()

        self.groupBox_1 = QGroupBox('Tabla de datos - %s' %self.strippedName(self.curFile), self.centralwidget)
        self.gb_1_HLyt = QHBoxLayout(self.groupBox_1)
        self.model = pandasModel(self.df.round({'X':4, 'Y':4, 'Z':4}))
        self.tableView = QTableView(self.groupBox_1)
        self.tableView.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableView.horizontalHeader().setStretchLastSection(True )
        self.tableView.setModel(self.model)

        self.gb_1_HLyt.addWidget(self.tableView)
        self.gb_1_2_VLyt.addWidget(self.groupBox_1)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.gb2_HLyt = QHBoxLayout(self.groupBox_2)

        self.horizontalSpacer = QSpacerItem(61, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb2_HLyt.addItem(self.horizontalSpacer)

        self.pushButton = QPushButton('OK', self.groupBox_2)
        self.pushButton.setShortcut("Return")
        self.pushButton.clicked.connect(okButton)
        self.gb2_HLyt.addWidget(self.pushButton)

        self.horizontalSpacer_2 = QSpacerItem(61, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb2_HLyt.addItem(self.horizontalSpacer_2)

        self.pushButton_2 = QPushButton('Cancelar', self.groupBox_2)
        self.pushButton_2.setShortcut('Escape')
        self.pushButton_2.clicked.connect(cancelButton)
        self.gb2_HLyt.addWidget(self.pushButton_2)

        self.horizontalSpacer_3 = QSpacerItem(61, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb2_HLyt.addItem(self.horizontalSpacer_3)

        self.gb_1_2_VLyt.addWidget(self.groupBox_2)

        self.cwHLyt.addLayout(self.gb_1_2_VLyt)

        self.groupBox_3 = QGroupBox('Grafica Aceleracion vs Tiempo', self.centralwidget)
        self.gb3_HLyt = QHBoxLayout(self.groupBox_3)
        genGraphs()
        self.gb3_HLyt.addWidget(self.figwindow)
        self.cwHLyt.addWidget(self.groupBox_3)

        self.cwHLyt.setStretch(0, 40)
        self.cwHLyt.setStretch(1, 60)

        self.setCentralWidget(self.centralwidget)

    def viewBaseLine(self):
        def changeComboBox():
            combotex = self.comboBox.currentText()

            if combotex == 'Polinomial':
                self.label_3.setVisible(False)
                self.lineEdit_2.setVisible(False)
            else:
                self.label_3.setVisible(True)
                self.lineEdit_2.setVisible(True)            

        def genGraphs():

            self.vel_corr = [ integrate.cumtrapz(self.acc_corr[i], dx=self.dt, initial=0.0) for i in range(3)]
            self.dsp_corr = [ integrate.cumtrapz(self.vel_corr[i], dx=self.dt, initial=0.0) for i in range(3)]

            max_acc = max([ np.max(np.abs(self.acc_corr[i])) for i in range(3)] )
            max_vel = max([ np.max(np.abs(self.vel_corr[i])) for i in range(3)] )
            max_dsp = max([ np.max(np.abs(self.dsp_corr[i])) for i in range(3)] )

            colors = ['b', 'g', 'k']
            direct = ['X', 'Y', 'Z']
            w = 0.5
            lbsize = 5.2
            ftsize = 'xx-small'

            for i in range(3):
                self.a[i].plot(self.t, self.acc_corr[i], colors[i], lw = w , label= 'pico: ' + str(round(np.max(np.abs(self.acc_corr[i])), 2)) + ' cm/s^2')
                self.a[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize= ftsize)
                self.a[i].set_ylabel(ylabel='Aceleración en %s ($cm/s^2$)' %direct[i], fontsize=ftsize)
                self.a[i].xaxis.set_tick_params(labelsize=lbsize)
                self.a[i].yaxis.set_tick_params(labelsize=lbsize)
                self.a[i].legend(loc='upper right', frameon=True, fontsize=ftsize, handlelength=2.0)
                self.a[i].label_outer()
                self.a[i].set_xlim(self.t[0], self.t[-1])
                self.a[i].set_ylim(-max_acc*1.05 , max_acc*1.05)
                self.a[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

                self.v[i].plot(self.t, self.vel_corr[i], colors[i], lw = w , label= 'pico: ' + str(round(np.max(np.abs(self.vel_corr[i])), 2)) + ' cm/s')
                self.v[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize=ftsize)
                self.v[i].set_ylabel(ylabel='Velocidad en %s ($cm/s$)' %direct[i], fontsize=ftsize)
                self.v[i].xaxis.set_tick_params(labelsize=lbsize)
                self.v[i].yaxis.set_tick_params(labelsize=lbsize)
                self.v[i].legend(loc='upper right', frameon=True, fontsize=ftsize, handlelength=2.0)
                self.v[i].label_outer()
                self.v[i].set_xlim(self.t[0], self.t[-1])
                self.v[i].set_ylim(-max_vel*1.05 , max_vel*1.05)
                self.v[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

                self.d[i].plot(self.t, self.dsp_corr[i], colors[i], lw = w , label= 'pico: ' + str(round(np.max(np.abs(self.dsp_corr[i])), 2)) + ' cm')
                self.d[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize=ftsize)
                self.d[i].set_ylabel(ylabel='Velocidad en %s ($cm/s$)' %direct[i], fontsize=ftsize)
                self.d[i].xaxis.set_tick_params(labelsize=lbsize)
                self.d[i].yaxis.set_tick_params(labelsize=lbsize)
                self.d[i].legend(loc='upper right', frameon=True, fontsize=ftsize, handlelength=2.0)
                self.d[i].label_outer()
                self.d[i].set_xlim(self.t[0], self.t[-1])
                self.d[i].set_ylim(-max_dsp*1.05 , max_dsp*1.05)
                self.d[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
            
            for i in range(3):
                self.figs[i].subplots_adjust(left=0.15, bottom=0.085, right=0.97, top=0.97)
                self.canvs[i].draw()   

        def apliButton():
            for i in range(3):
                del self.a[i].lines[:]
                del self.v[i].lines[:]
                del self.d[i].lines[:]

            kind = self.comboBox.currentText()
            order = int(self.lineEdit_1.text())
            spline = int(self.lineEdit_2.text())
 
            self.acc_corr = [ BaseLineCorrection(self.acc[i], dt=self.dt, type=kind, order=order, dspline=spline)  for i in range(3)]
            genGraphs()

        def okButton():
            self.acc = copy(self.acc_corr)
            self.centralwidget.deleteLater()
            self.viewStart()

        def cancelButton():
            self.acc_corr = copy(self.acc)
            self.centralwidget.deleteLater()
            self.viewStart()

        self.figs = [ Figure() for i in range(3) ]
        self.canvs = [ FigureCanvas(self.figs[i]) for i in range(3) ]
        self.a = self.canvs[0].figure.subplots(3)
        self.v = self.canvs[1].figure.subplots(3)
        self.d = self.canvs[2].figure.subplots(3)


        self.centralwidget = QWidget(self)
        self.cw_VLyt = QVBoxLayout(self.centralwidget)

        self.gb_1_2_3_HLyt = QHBoxLayout()

        self.groupBox_1 = QGroupBox('Aceleracion', self.centralwidget)
        self.groupBox_1.setAlignment(Qt.AlignCenter)
        self.gb_1_HLyt = QHBoxLayout(self.groupBox_1)
        wa = QMainWindow()
        waWidget = QWidget()
        waLayout = QHBoxLayout(waWidget)
        waLayout.addWidget(self.canvs[0])
        wa.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(self.canvs[0], self))
        wa.setCentralWidget(waWidget)
        self.gb_1_HLyt.addWidget(wa)

        self.gb_1_2_3_HLyt.addWidget(self.groupBox_1)

        self.groupBox_2 = QGroupBox('Velocidad', self.centralwidget)
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt = QHBoxLayout(self.groupBox_2)
        wv = QMainWindow()
        wvWidget = QWidget()
        wvLayout = QHBoxLayout(wvWidget)
        wvLayout.addWidget(self.canvs[1])
        wv.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(self.canvs[1], self))
        wv.setCentralWidget(wvWidget)
        self.gb_2_HLyt.addWidget(wv)

        self.gb_1_2_3_HLyt.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox('Desplazamiento', self.centralwidget)
        self.groupBox_3.setAlignment(Qt.AlignCenter)
        self.gb_3_HLyt = QHBoxLayout(self.groupBox_3)
        wd = QMainWindow()
        wdWidget = QWidget()
        wdLayout = QHBoxLayout(wdWidget)
        wdLayout.addWidget(self.canvs[2])
        wd.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(self.canvs[2], self))
        wd.setCentralWidget(wdWidget)
        self.gb_3_HLyt.addWidget(wd)

        self.gb_1_2_3_HLyt.addWidget(self.groupBox_3)

        self.cw_VLyt.addLayout(self.gb_1_2_3_HLyt)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.gb_4_HLyt = QHBoxLayout(self.groupBox_4)

        self.horizontalSpacer_1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_4_HLyt.addItem(self.horizontalSpacer_1)  

        self.label_1 = QLabel('Tipo:', self.groupBox_4) 
        self.label_1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_4_HLyt.addWidget(self.label_1)   

        self.comboBox = QComboBox(self.groupBox_4)
        self.comboBox.addItem("Spline")
        self.comboBox.addItem("Polinomial")
        self.gb_4_HLyt.addWidget(self.comboBox)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_4_HLyt.addItem(self.horizontalSpacer_2)

        self.label_2 = QLabel('Orden:', self.groupBox_4)
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_4_HLyt.addWidget(self.label_2)

        self.lineEdit_1 = QLineEdit('1', self.groupBox_4)
        self.lineEdit_1.setAlignment(Qt.AlignCenter)
        self.gb_4_HLyt.addWidget(self.lineEdit_1)

        self.label_3 = QLabel('N° puntos:', self.groupBox_4)
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_4_HLyt.addWidget(self.label_3)
    
        self.lineEdit_2 = QLineEdit('1000', self.groupBox_4)
        self.lineEdit_2.setAlignment(Qt.AlignCenter)
        self.gb_4_HLyt.addWidget(self.lineEdit_2)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_4_HLyt.addItem(self.horizontalSpacer_3)

        self.pushButton_1 = QPushButton('Aplicar', self.groupBox_4)
        self.pushButton_1.clicked.connect(apliButton)
        self.gb_4_HLyt.addWidget(self.pushButton_1)

        self.horizontalSpacer_4 = QSpacerItem(70, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_4_HLyt.addItem(self.horizontalSpacer_4)

        self.pushButton_2 = QPushButton('OK', self.groupBox_4)
        self.pushButton_2.setShortcut("Return")
        self.pushButton_2.clicked.connect(okButton)
        self.gb_4_HLyt.addWidget(self.pushButton_2)

        self.pushButton_3 = QPushButton('Cancelar', self.groupBox_4)
        self.pushButton_3.setShortcut("Escape")
        self.pushButton_3.clicked.connect(cancelButton)
        self.gb_4_HLyt.addWidget(self.pushButton_3)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_4_HLyt.addItem(self.horizontalSpacer_5)

        self.cw_VLyt.addWidget(self.groupBox_4)

        genGraphs()

        self.gb_4_HLyt.setStretch(0, 2)
        self.gb_4_HLyt.setStretch(1, 2)
        self.gb_4_HLyt.setStretch(2, 5)
        self.gb_4_HLyt.setStretch(3, 1)
        self.gb_4_HLyt.setStretch(4, 2)
        self.gb_4_HLyt.setStretch(5, 2)
        self.gb_4_HLyt.setStretch(6, 2)
        self.gb_4_HLyt.setStretch(7, 2)
        self.gb_4_HLyt.setStretch(8, 2)
        self.gb_4_HLyt.setStretch(9, 5)
        self.gb_4_HLyt.setStretch(10, 5)
        self.gb_4_HLyt.setStretch(11, 5)
        self.gb_4_HLyt.setStretch(12, 5)
        self.gb_4_HLyt.setStretch(13, 5)

        self.setCentralWidget(self.centralwidget)

        self.comboBox.textActivated.connect(changeComboBox)

    def viewPassBand(self):
        def apliButton():
            
            for i in range(3):
                del self.a[i].lines[:]
            del self.f.lines[:]

            n = float(self.lineEdit_1.text())
            fl = float(self.lineEdit_2.text())
            fh = float(self.lineEdit_3.text())

            self.acc_corr = [ Butterworth_Bandpass(self.acc[i], self.dt, fl, fh, n) for i in range(3)]
            genGraphs(vlines=True)

        def okButton():
            self.acc = copy(self.acc_corr)
            self.centralwidget.deleteLater()
            self.viewStart()

        def cancelButton():
            self.acc_corr = copy(self.acc)
            self.centralwidget.deleteLater()
            self.viewStart()

        def genGraphs(vlines=False):

            self.fou = [np.fft.rfft(self.acc_corr[i]) for i in range(3)]
            self.fre = np.fft.rfftfreq(len(self.t), d = self.dt)

            max_acc = max([ np.max(np.abs(self.acc_corr[i])) for i in range(3)] )
            max_fou = max([ np.max(np.abs(self.fou[i])/self.t[-1]) for i in range(3)] )

            colors = ['b', 'g', 'k']
            direct = ['X', 'Y', 'Z']
            w = 0.5
            lbsize = 5.2
            ftsize = 'xx-small'

            for i in range(3):
                self.f.plot(self.fre, np.abs(self.fou[i])/self.t[-1], colors[i], lw = w , label= 'pico: ' + str(round(np.max(np.abs(self.fou[i])/self.t[-1]), 2)) + ' cm/s')
            self.f.set_xlabel(xlabel='$Frecuencia (Hz)$', fontsize= ftsize)
            self.f.set_ylabel(ylabel='Amplitus de Fourier %s ($cm/s$)' %direct[i], fontsize=ftsize)
            self.f.xaxis.set_tick_params(labelsize=lbsize)
            self.f.yaxis.set_tick_params(labelsize=lbsize)
            self.f.legend(loc='upper right', frameon=True, fontsize=ftsize, handlelength=2.0)
            self.f.set_xlim(self.fre[0], self.fre[-1])
            self.f.set_ylim(0.0, max_fou*1.05)
            self.f.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

            if vlines:
                self.f.axvline(x = float(self.lineEdit_2.text()), ymin = 0, ymax = ceil(max_fou), color='m', lw=1)
                self.f.axvline(x = float(self.lineEdit_3.text()), ymin = 0, ymax = ceil(max_fou), color='m', lw=1)


            self.figs[0].subplots_adjust(left=0.1, bottom=0.085, right=0.97, top=0.97)
            self.canvs[0].draw()  

            for i in range(3):
                self.a[i].plot(self.t, self.acc_corr[i], colors[i], lw = w , label= 'pico: ' + str(round(np.max(np.abs(self.acc_corr[i])), 2)) + ' cm/s^2')
                self.a[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize= ftsize)
                self.a[i].set_ylabel(ylabel='Aceleración en %s ($cm/s^2$)' %direct[i], fontsize=ftsize)
                self.a[i].xaxis.set_tick_params(labelsize=lbsize)
                self.a[i].yaxis.set_tick_params(labelsize=lbsize)
                self.a[i].legend(loc='upper right', frameon=True, fontsize=ftsize, handlelength=2.0)
                self.a[i].label_outer()
                self.a[i].set_xlim(self.t[0], self.t[-1])
                self.a[i].set_ylim(-max_acc*1.05 , max_acc*1.05)
                self.a[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

            self.figs[1].subplots_adjust(left=0.1, bottom=0.085, right=0.97, top=0.97)
            self.canvs[1].draw()  


        self.figs = [ Figure() for i in range(2) ]
        self.canvs = [ FigureCanvas(self.figs[i]) for i in range(2) ]
        self.f = self.canvs[0].figure.subplots(1)
        self.a = self.canvs[1].figure.subplots(3)

        genGraphs()

        self.centralwidget = QWidget(self)
        self.cw_VLyt = QVBoxLayout(self.centralwidget)

        self.gb_1_2_HLyt = QHBoxLayout()

        self.groupBox_1 = QGroupBox('Espectros de Fourier', self.centralwidget)
        self.groupBox_1.setAlignment(Qt.AlignCenter)
        self.gb_1_HLyt = QHBoxLayout(self.groupBox_1)

        wf = QMainWindow()
        wfWidget = QWidget()
        wfLayout = QHBoxLayout(wfWidget)
        wfLayout.addWidget(self.canvs[0])
        wf.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(self.canvs[0], self))
        wf.setCentralWidget(wfWidget)
        self.gb_1_HLyt.addWidget(wf)

        self.gb_1_2_HLyt.addWidget(self.groupBox_1)

        self.groupBox_2 = QGroupBox('Aceleraciones', self.centralwidget)
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt = QHBoxLayout(self.groupBox_2)

        wa = QMainWindow()
        waWidget = QWidget()
        waLayout = QHBoxLayout(waWidget)
        waLayout.addWidget(self.canvs[1])
        wa.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(self.canvs[1], self))
        wa.setCentralWidget(waWidget)
        self.gb_2_HLyt.addWidget(wa)

        self.gb_1_2_HLyt.addWidget(self.groupBox_2)

        self.cw_VLyt.addLayout(self.gb_1_2_HLyt)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.gb_3_HLyt = QHBoxLayout(self.groupBox_3)
        self.horizontalSpacer_1 = QSpacerItem(37, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gb_3_HLyt.addItem(self.horizontalSpacer_1)

        self.label_1 = QLabel('Orden:', self.groupBox_3)
        self.label_1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_3_HLyt.addWidget(self.label_1)

        self.lineEdit_1 = QLineEdit('5', self.groupBox_3)
        self.lineEdit_1.setAlignment(Qt.AlignCenter)

        self.gb_3_HLyt.addWidget(self.lineEdit_1)

        self.label_2 = QLabel('Low Cut (Hz)', self.groupBox_3)
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_3_HLyt.addWidget(self.label_2)

        self.lineEdit_2 = QLineEdit('0.1', self.groupBox_3)
        self.lineEdit_2.setAlignment(Qt.AlignCenter)
        self.gb_3_HLyt.addWidget(self.lineEdit_2)

        self.label_3 = QLabel('High Cut (Hz)', self.groupBox_3)
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_3_HLyt.addWidget(self.label_3)

        self.lineEdit_3 = QLineEdit('20', self.groupBox_3)
        self.lineEdit_3.setAlignment(Qt.AlignCenter)
        self.gb_3_HLyt.addWidget(self.lineEdit_3)

        self.horizontalSpacer_2 = QSpacerItem(17, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_3_HLyt.addItem(self.horizontalSpacer_2)

        self.pushButton_1 = QPushButton('Aplicar', self.groupBox_3)
        self.pushButton_1.clicked.connect(apliButton)

        self.gb_3_HLyt.addWidget(self.pushButton_1)

        self.horizontalSpacer_3 = QSpacerItem(96, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_3_HLyt.addItem(self.horizontalSpacer_3)

        self.pushButton_2 = QPushButton('OK', self.groupBox_3)
        self.pushButton_2.setShortcut("Return")
        self.pushButton_2.clicked.connect(okButton)

        self.gb_3_HLyt.addWidget(self.pushButton_2)

        self.pushButton_3 = QPushButton('Cancelar', self.groupBox_3)
        self.pushButton_3.setShortcut("Escape")
        self.pushButton_3.clicked.connect(cancelButton)
        self.gb_3_HLyt.addWidget(self.pushButton_3)

        self.horizontalSpacer_4 = QSpacerItem(96, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_3_HLyt.addItem(self.horizontalSpacer_4)

        self.gb_3_HLyt.setStretch(0, 2)
        self.gb_3_HLyt.setStretch(1, 2)
        self.gb_3_HLyt.setStretch(2, 2)
        self.gb_3_HLyt.setStretch(3, 3)
        self.gb_3_HLyt.setStretch(4, 2)
        self.gb_3_HLyt.setStretch(5, 3)
        self.gb_3_HLyt.setStretch(6, 2)
        self.gb_3_HLyt.setStretch(7, 1)
        self.gb_3_HLyt.setStretch(8, 5)
        self.gb_3_HLyt.setStretch(9, 5)
        self.gb_3_HLyt.setStretch(10, 5)
        self.gb_3_HLyt.setStretch(11, 5)
        self.gb_3_HLyt.setStretch(12, 5)

        self.cw_VLyt.addWidget(self.groupBox_3)

        self.setCentralWidget(self.centralwidget)

    def viewSimula(self):

        def mdof(n, direct='X', m=10000, k=2000000):


            if direct == 'X':
                self.at = self.acc_corr[0]
            else:
                self.at = self.acc_corr[1]
            # m = 10000 # Kg
            # k = 20000000 # Kgf/cm
            self.mdof = VGL()
            mm = self.mdof.MatrizMasa([m for i in range(n)])
            kk = self.mdof.MatrizRigidez([k for i in range(n)])
            self.mdof.Modos(500)

            I = np.ones((len(mm[0]),1))
            p = -mm@I*self.at
            self.mdof.Newmark(n, p , self.dt)

        def animate(step, factor):
            self.alt = 0.0
            self.h = float(self.lineEdit_4.text())
            for i in range(self.n_floor):
                j = (self.n_floor - 1) - i
                self.line_acc[i].set_xdata(self.t[:step])
                self.line_acc[i].set_ydata(self.mdof.upp[j][:step])
                self.line_vel[i].set_xdata(self.t[:step])
                self.line_vel[i].set_ydata(self.mdof.up[j][:step])
                self.line_dsp[i].set_xdata(self.t[:step])
                self.line_dsp[i].set_ydata(self.mdof.u[j][:step])

            self.line_acc[-1].set_xdata(self.t[:step])
            self.line_acc[-1].set_ydata(self.at[:step])
            self.line_vel[-1].set_xdata(self.t[:step])
            self.line_vel[-1].set_ydata(self.upt[:step])
            self.line_dsp[-1].set_xdata(self.t[:step])
            self.line_dsp[-1].set_ydata(self.ut[:step])

            amp = float(self.lineEdit_5.text())
                
            self.line_lc[0].set_data([-self.xx, -self.xx + amp*self.mdof.u[0][step]], [self.yy[0][step], self.yy[1][step]])
            self.line_rc[0].set_data([ self.xx,  self.xx + amp*self.mdof.u[0][step]], [self.yy[0][step], self.yy[1][step]])
            self.line_fl[0].set_data([-self.xx + amp*self.mdof.u[0][step], self.xx + amp*self.mdof.u[0][step]], [self.yy[1][step], self.yy[1][step]])

            for i in range(1, self.n_floor):
                self.alt += self.h
                self.line_lc[i].set_data([-self.xx + amp*self.mdof.u[i-1][step], -self.xx + amp*self.mdof.u[i][step]], [self.yy[i][step], self.yy[i+1][step]])
                self.line_rc[i].set_data([ self.xx + amp*self.mdof.u[i-1][step],  self.xx + amp*self.mdof.u[i][step]], [self.yy[i][step], self.yy[i+1][step]])
                self.line_fl[i].set_data([-self.xx + amp*self.mdof.u[i][step], self.xx + amp*self.mdof.u[i][step]], [self.yy[i+1][step], self.yy[i+1][step]])

            r = self.line_acc + self.line_vel + self.line_dsp + self.line_lc + self.line_rc + self.line_fl

            return r
        
        def genGraphs(play=False):
            w = 0.5
            colors = ['b', 'g', 'k']
            alpha = 0.2
            
            self.acc_limit = max([np.max(np.abs(self.mdof.upp[i])) for i in range(self.n_floor)])
            self.vel_limit = max([np.max(np.abs(self.mdof.up[i])) for i in range(self.n_floor)])
            self.dsp_limit = max([np.max(np.abs(self.mdof.u[i])) for i in range(self.n_floor)])

            self.ax_dsp = []
            self.ax_vel = []
            self.ax_acc = []

            self.upt = integrate.cumtrapz(self.at, dx=self.dt, initial=0.0)
            self.ut = integrate.cumtrapz(self.upt, dx=self.dt, initial=0.0)
            self.acc_limit = max([ self.acc_limit, np.max(np.abs(self.at))])
            self.vel_limit = max([ self.vel_limit, np.max(np.abs(self.upt))])
            self.dsp_limit = max([ self.dsp_limit, np.max(np.abs(self.ut))])

            for i in range(self.n_floor):
                self.ax_acc.append(self.canvs.figure.add_subplot(self.gs[i,0]))
                self.ax_vel.append(self.canvs.figure.add_subplot(self.gs[i,1]))
                self.ax_dsp.append(self.canvs.figure.add_subplot(self.gs[i,2]))
            self.ax_acc.append(self.canvs.figure.add_subplot(self.gs[-1,0]))
            self.ax_vel.append(self.canvs.figure.add_subplot(self.gs[-1,1]))
            self.ax_dsp.append(self.canvs.figure.add_subplot(self.gs[-1,2]))        
            
            self.bld = self.canvs.figure.add_subplot(self.gs[:,3])

            for i in range(self.n_floor):
                j = (self.n_floor - 1) - i
                self.ax_acc[i].plot(self.t, self.mdof.upp[j], colors[0],  lw=w, alpha=alpha, label='Piso %d - pico: '%(j+1) + str(round(np.max(np.abs(self.mdof.upp[j])) , 2)) )
                self.ax_acc[i].xaxis.set_tick_params(labelsize=6)
                self.ax_acc[i].yaxis.set_tick_params(labelsize=6)
                self.ax_acc[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'xx-small')
                self.ax_acc[i].set_ylabel(ylabel='', fontsize= 'xx-small')
                self.ax_acc[i].legend(loc='upper right', frameon=True, fontsize= 'xx-small', handlelength=2.0)
                self.ax_acc[i].xaxis.set_ticklabels([])
                self.ax_acc[i].set_xlim(self.t[0], self.t[-1])
                self.ax_acc[i].set_ylim(-self.acc_limit*1.05 , self.acc_limit*1.05)
                self.ax_acc[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                if i == 0:
                    self.ax_acc[i].set_title('Aceleracion (cm/s2)', fontsize= 'xx-small')

                self.ax_vel[i].plot(self.t, self.mdof.up[j], colors[1],  lw = w, alpha=alpha, label='Piso %d - pico: '%(j+1) + str(round(np.max(np.abs(self.mdof.up[j])) , 2)))
                self.ax_vel[i].xaxis.set_tick_params(labelsize=6)
                self.ax_vel[i].yaxis.set_tick_params(labelsize=6)
                self.ax_vel[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'xx-small')
                self.ax_vel[i].set_ylabel(ylabel='', fontsize= 'xx-small')
                self.ax_vel[i].legend(loc='upper right', frameon=True, fontsize= 'xx-small', handlelength=2.0)
                self.ax_vel[i].xaxis.set_ticklabels([])
                self.ax_vel[i].set_xlim(self.t[0], self.t[-1])  
                self.ax_vel[i].set_ylim(-self.vel_limit*1.05 , self.vel_limit*1.05)
                self.ax_vel[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                if i == 0:
                    self.ax_vel[i].set_title('Velocidad (cm/s)', fontsize= 'xx-small')

                self.ax_dsp[i].plot(self.t, self.mdof.u[j], colors[2],  lw = w, alpha=alpha, label='Piso %d - pico: '%(j+1) + str(round(np.max(np.abs(self.mdof.u[j])) , 2)))
                self.ax_dsp[i].xaxis.set_tick_params(labelsize=6)
                self.ax_dsp[i].yaxis.set_tick_params(labelsize=6)
                self.ax_dsp[i].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'xx-small')
                self.ax_dsp[i].set_ylabel(ylabel='', fontsize= 'xx-small')
                self.ax_dsp[i].legend(loc='upper right', frameon=True, fontsize= 'xx-small', handlelength=2.0)
                self.ax_dsp[i].xaxis.set_ticklabels([])
                self.ax_dsp[i].set_xlim(self.t[0], self.t[-1]) 
                self.ax_dsp[i].set_ylim(-self.dsp_limit*1.05 , self.dsp_limit*1.05)
                self.ax_dsp[i].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                if i == 0:
                    self.ax_dsp[i].set_title('Desplazamiento (cm)', fontsize= 'xx-small')

            self.ax_acc[-1].plot(self.t, self.at, colors[0],  lw = w , alpha=alpha, label='Terreno - pico: '+str(round(np.max(np.abs(self.at)), 2)))
            self.ax_acc[-1].xaxis.set_tick_params(labelsize=6)
            self.ax_acc[-1].yaxis.set_tick_params(labelsize=6)
            self.ax_acc[-1].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'xx-small')
            self.ax_acc[-1].set_ylabel(ylabel='', fontsize= 'xx-small')
            self.ax_acc[-1].legend(loc='upper right', frameon=True, fontsize= 'xx-small', handlelength=2.0)
            self.ax_acc[-1].set_xlim(self.t[0], self.t[-1])
            self.ax_acc[-1].set_ylim(-self.acc_limit*1.05 , self.acc_limit*1.05)
            self.ax_acc[-1].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

            self.ax_vel[-1].plot(self.t, self.upt, colors[1],  lw = w, alpha=alpha, label='Terreno - pico: '+str(round(np.max(np.abs(self.upt)), 2)))
            self.ax_vel[-1].xaxis.set_tick_params(labelsize=6)
            self.ax_vel[-1].yaxis.set_tick_params(labelsize=6)
            self.ax_vel[-1].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'xx-small')
            self.ax_vel[-1].set_ylabel(ylabel='', fontsize= 'xx-small')
            self.ax_vel[-1].legend(loc='upper right', frameon=True, fontsize= 'xx-small', handlelength=2.0)
            self.ax_vel[-1].set_xlim(self.t[0], self.t[-1])  
            self.ax_vel[-1].set_ylim(-self.vel_limit*1.05 , self.vel_limit*1.05)
            self.ax_vel[-1].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)

            self.ax_dsp[-1].plot(self.t, self.ut, colors[2],  lw = w, alpha=alpha, label='Terreno - pico: '+str(round(np.max(np.abs(self.ut)), 2)))
            self.ax_dsp[-1].xaxis.set_tick_params(labelsize=6)
            self.ax_dsp[-1].yaxis.set_tick_params(labelsize=6)
            self.ax_dsp[-1].set_xlabel(xlabel='$Tiempo (s)$', fontsize= 'xx-small')
            self.ax_dsp[-1].set_ylabel(ylabel='', fontsize= 'xx-small')
            self.ax_dsp[-1].legend(loc='upper right', frameon=True, fontsize= 'xx-small', handlelength=2.0)
            self.ax_dsp[-1].set_xlim(self.t[0], self.t[-1])
            self.ax_dsp[-1].set_ylim(-self.dsp_limit*1.05 , self.dsp_limit*1.05)
            self.ax_dsp[-1].grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)


            self.xx = 50*float(self.lineEdit_3.text())
            self.h = float(self.lineEdit_4.text())
            self.alt = 0.0
            offset = 200.0
            for i in range(self.n_floor):
                self.alt += self.h
                self.bld.plot([-self.xx, -self.xx], [self.alt-self.h, self.alt], 'k', lw=5, alpha=alpha) # col izquierda
                self.bld.plot([self.xx, self.xx], [self.alt-self.h, self.alt], 'k', lw=5, alpha=alpha) # col derecha
                self.bld.plot([-self.xx, self.xx], [self.alt, self.alt], 'k', lw=8, alpha=alpha) # techo
            self.bld.xaxis.set_tick_params(labelsize=6)
            self.bld.yaxis.set_tick_params(labelsize=6)
            self.bld.set_xlabel(xlabel='$cm$', fontsize= 'xx-small')
            
            self.bld.axvline(x = 0, ymin = 0, ymax = self.alt + 0.1*self.h, color='k', lw=w)
            self.bld.set_xlim(-(self.xx+offset), self.xx+offset)
            self.bld.set_ylim(0, self.alt + 0.2*self.h)

            if play:
                self.line_acc = []
                self.line_vel = []
                self.line_dsp = []
                self.line_lc = []
                self.line_rc = []
                self.line_fl = []

                self.alt = 0.0
                self.yy = []

                self.lc, = self.bld.plot([], [],'k-',lw=5)
                self.rc, = self.bld.plot([], [],'k-',lw=5)
                self.fl, = self.bld.plot([], [],'k-',lw=8)

                for i in range(self.n_floor+1):
                    a, = self.ax_acc[i].plot([], [], colors[0], lw=w)
                    self.line_acc.append(a)
                    v, = self.ax_vel[i].plot([], [], colors[1], lw=w)
                    self.line_vel.append(v)
                    d, = self.ax_dsp[i].plot([], [], colors[2], lw=w)
                    self.line_dsp.append(d)
                    lc, = self.bld.plot([], [],'k-',lw=5)
                    self.line_lc.append(lc)
                    rc, = self.bld.plot([], [],'k-',lw=5)
                    self.line_rc.append(rc)
                    fl, = self.bld.plot([], [],'k-',lw=8)
                    self.line_fl.append(fl)
                    self.yy.append(np.array([self.alt for i in range(len(self.t))]))
                    self.alt += self.h
                    
                factor = 1000
                ani = animation.FuncAnimation(self.fig, animate, frames=len(self.t), fargs=(factor,), interval=0.1, blit=True)

            self.fig.subplots_adjust(left=0.035, bottom=0.07, right=0.985, top=0.95,  hspace=0.0, wspace=0.15)
            self.canvs.draw()
            
        def playButton():

            for i in range(len(self.ax_acc)):
                 self.ax_acc[i].remove
                 self.ax_vel[i].remove
                 self.ax_dsp[i].remove

            self.fig.clf()

            self.n_floor = int(self.comboBox_2.currentText())
            m = 1000*float(self.lineEdit_1.text())
            k = 1000*float(self.lineEdit_2.text())
            
            mdof(self.n_floor, direct=self.comboBox_1.currentText(), m=m, k=k)

            self.gs = self.fig.add_gridspec(self.n_floor+1, 4)

            genGraphs(play=True)

        def resetButton():
            for i in range(len(self.ax_acc)):
                 self.ax_acc[i].remove
                 self.ax_vel[i].remove
                 self.ax_dsp[i].remove

            self.fig.clf()

            self.n_floor = int(self.comboBox_2.currentText())
            m = 1000*float(self.lineEdit_1.text())
            k = 1000*float(self.lineEdit_2.text())
            
            mdof(self.n_floor, direct=self.comboBox_1.currentText(), m=m, k=k)

            self.gs = self.fig.add_gridspec(self.n_floor+1, 4)

            genGraphs(play=False)

        def closeButton():
            resetButton()
            self.centralwidget.deleteLater()
            self.viewStart()

        self.n_floor = 4
        mdof(self.n_floor, direct='X')

        self.fig = Figure()
        self.gs = self.fig.add_gridspec(self.n_floor+1, 4)
        self.canvs = FigureCanvas(self.fig)

        self.centralwidget = QWidget(self)

        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.groupBox = QGroupBox('Simulacion', self.centralwidget)
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.gb_1_HLyt = QHBoxLayout(self.groupBox)
        w = QMainWindow()
        wWidget = QWidget()
        wLayout = QHBoxLayout(wWidget)
        wLayout.addWidget(self.canvs)
        w.addToolBar(Qt.BottomToolBarArea, NavigationToolbar2QT(self.canvs, self))
        w.setCentralWidget(wWidget)
        self.gb_1_HLyt.addWidget(w)

        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox('Opciones', self.centralwidget)
        self.gb_2_HLyt = QHBoxLayout(self.groupBox_2)
        self.horizontalSpacer_1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gb_2_HLyt.addItem(self.horizontalSpacer_1)

        self.label_1 = QLabel('Direccion:', self.groupBox_2)
        self.label_1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_1)

        self.comboBox_1 = QComboBox(self.groupBox_2)
        self.comboBox_1.addItem("X")
        self.comboBox_1.addItem("Y")
        self.gb_2_HLyt.addWidget(self.comboBox_1)

        self.label_2 = QLabel('N° Pisos:', self.groupBox_2)
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_2)

        self.comboBox_2 = QComboBox(self.groupBox_2)
        for i in range(2,11):
            self.comboBox_2.addItem("%d" %(i))
        self.comboBox_2.setCurrentText('4')
        self.gb_2_HLyt.addWidget(self.comboBox_2)
        
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_2_HLyt.addItem(self.horizontalSpacer_2)

        self.label_3 = QLabel('M (Tnf):', self.groupBox_2)
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_3)

        self.lineEdit_1 = QLineEdit('10', self.groupBox_2)
        self.lineEdit_1.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt.addWidget(self.lineEdit_1)


        self.label_4 = QLabel('K (Tnf/cm):', self.groupBox_2)
        self.label_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_4)

        self.lineEdit_2 = QLineEdit('2000', self.groupBox_2)
        self.lineEdit_2.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt.addWidget(self.lineEdit_2)

        self.label_5 = QLabel('B (m):', self.groupBox_2)
        self.label_5.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_5)

        self.lineEdit_3 = QLineEdit('5', self.groupBox_2)
        self.lineEdit_3.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt.addWidget(self.lineEdit_3)

        self.label_6 = QLabel('H (m):', self.groupBox_2)
        self.label_6.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_6)

        self.lineEdit_4 = QLineEdit('2.8', self.groupBox_2)
        self.lineEdit_4.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt.addWidget(self.lineEdit_4)

        self.label_7 = QLabel('Amplificacion:', self.groupBox_2)
        self.label_7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.gb_2_HLyt.addWidget(self.label_7)

        self.lineEdit_5 = QLineEdit('1.0', self.groupBox_2)
        self.lineEdit_5.setAlignment(Qt.AlignCenter)
        self.gb_2_HLyt.addWidget(self.lineEdit_5)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_2_HLyt.addItem(self.horizontalSpacer_3)

        self.pushButton_1 = QPushButton('Play', self.groupBox_2)
        self.pushButton_1.clicked.connect(playButton)
        self.gb_2_HLyt.addWidget(self.pushButton_1)

        self.pushButton_2 = QPushButton('Reset', self.groupBox_2)
        self.pushButton_2.clicked.connect(resetButton)
        self.gb_2_HLyt.addWidget(self.pushButton_2)

        self.pushButton_3 = QPushButton('Salir', self.groupBox_2)
        self.pushButton_3.clicked.connect(closeButton)
        self.gb_2_HLyt.addWidget(self.pushButton_3)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gb_2_HLyt.addItem(self.horizontalSpacer_4)

        self.verticalLayout.addWidget(self.groupBox_2)

        self.gb_2_HLyt.setStretch(1, 5)
        self.gb_2_HLyt.setStretch(2, 5)
        self.gb_2_HLyt.setStretch(3, 5)
        self.gb_2_HLyt.setStretch(4, 5)
        self.gb_2_HLyt.setStretch(6, 5)
        self.gb_2_HLyt.setStretch(7, 5)
        self.gb_2_HLyt.setStretch(8, 5)
        self.gb_2_HLyt.setStretch(9, 5)
        self.gb_2_HLyt.setStretch(10, 5)
        self.gb_2_HLyt.setStretch(11, 5)
        self.gb_2_HLyt.setStretch(12, 5)
        self.gb_2_HLyt.setStretch(13, 5)
        self.gb_2_HLyt.setStretch(15, 5)
        self.gb_2_HLyt.setStretch(16, 5)
        self.gb_2_HLyt.setStretch(17, 5)

        self.setCentralWidget(self.centralwidget)

        genGraphs()

##################################################################################

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class VGL:

	def __init__(self):
		"""
		"""
		pass

	def MatrizRigidez(self, args):
		"""
                Construye la matriz de rigideces pasando por parámetro una tupla con
		los valores de las rigideces.
		"""
		n = len(args)
		self.n = n
		self.k = np.zeros((n,n))

		self.k[0][0] = args[0] + args[1]
		self.k[0][1] = -args[1]

		for i in range(1,n-1):
			self.k[i][i-1] = -args[i]
			self.k[i][i] = args[i] + args[i+1]
			self.k[i][i+1] = -args[i+1]

		self.k[n-1][n-2] = -args[n-1]
		self.k[n-1][n-1] = args[n-1]

		return self.k

	def MatrizMasa(self, args):
		"""
		Construye la matriz de masas pasando por parámetro una tupla con
		los valores de las masas.
		"""
		n = len(args)
		self.n = n
		self.m = np.zeros((n,n))

		for i in range(n):
			self.m[i][i] = args[i]

		return self.m

	def Modos(self, iteraciones):

		# Comvirtiendo a la forma clásica
		r = np.zeros((self.n, self.n))

		for i in range(self.n):
			r[i][i] = self.m[i][i]**(-0.5)

		A = r@self.k@r

		jacobi = Jacobi(A, iteraciones)
		self.T = 2*np.pi*jacobi.Ω.diagonal()**(-1)
		self.Ω = jacobi.Ω
		self.Φ = jacobi.Φ

		# Normalizando los modos
		for i in range(self.n):
			self.Φ[:,i:i+1] = self.Φ[:,i:i+1]/(self.Φ[:,i:i+1].T@self.m@self.Φ[:,i:i+1])**0.5
		
		
		# Ordenando de mayor a menor periodo del modo (Burbuja)
		G = copy(self.Φ.T)
		for i in range(self.n-1):
			for j in range(i+1,self.n):
				if self.T[i] < self.T[j]:
					temp1, temp2, temp3  = self.T[i], self.Ω[i][i], copy(G[i])
					self.T[i], self.Ω[i][i], G[i] = self.T[j], self.Ω[j][j], copy(G[j])
					self.T[j], self.Ω[j][j], G[j] = temp1, temp2, temp3
		self.Φ = G.T

		# Factores de participación estática
		self.Γ = np.zeros(self.n)
		I = np.ones((self.n,1))
		for i in range(self.n):
			x = self.Φ[:,i:i+1].T@self.m@I/(self.Φ[:,i:i+1].T@self.m@self.Φ[:,i:i+1])
			self.Γ[i] = x[0][0]

	def Newmark(self , J , p , Δt , ζ = 0.05 , β = 1/4 , γ = 1/2):
		"""
		Resuelve el sistema de ecuaciones diferenciales de un sistema VGL de forma matricial a traves del método de Newmark
		El sistema de ecuaciones tiene ma forma:
				m*upp + c*up + k*u = p(t)  Para exitaciones sísmicas p(t) = -m*I*at(t)
		Dónde:
		m : matriz de masas
		k : matriz de rigideces
		c : matriz de coeficientes de fricción
		u : respuesta de desplazamientos de cada nivel relativo a la base.
		at(t): acelelacion del terreno 

		El sistema se transforma a coordenadas nodales según Chopra de la siguiente manera:
				M*qpp + C*qp + K*q = P(t)
		Dónde:
		M = ΦT*m*Φ
		C = ΦT*c*Φ
		K = ΦT*k*Φ
		P(t) = ΦT*p(t)
		Φ : Matriz modal

		Luego las respuestas del sistema original es:
				u(t) = Φ*q(t)
				upp(t) = Φ*qpp(t)

		Parámetros:
		J : Cantidad de modos a participar
		p : Para exitaciones sísmicas -m*I*at(t)
		Δt : Paso de tiempo de la aceleracion del terreno at(t) o de p(t)
		ζ : Fracción de amortiguamiendo modal, se considera que es igual para todos los modos
		γ : parametro de presición, generalmente 1/2
        β : razón de la variacion de la aceleración, generalmente entre 1/4 y 1/6
            Para β=1/6 se le llama el método de la aceleración lineal y para
            β=1/4, método de la aceleración de promedio constante.
        	El método es convergente si Δt/Tn < (1/π√2)[1/√(γ −2β)] donde Tn es el perido del modo n.
		"""

		# print(J, p, Δt)
		# print(type(J), type(p),type(Δt))
		m = self.m
		k = self.k
		Φ = self.Φ[:,0:J]
		Ω = self.Ω[0:J,0:J]

		M = Φ.T@m@Φ
		K = Φ.T@k@Φ
		C = 2*ζ*M@Ω
		
		n = len(M[0])
		m = len(p[0])

		# 1.1) Se considera que el sistema parte del reposo
		q = np.zeros((n,m))
		qp = np.zeros((n,m))
		# 1.2) # P[0] = Φ.T@p[0]
		P0 = Φ.T@p[:,0:1]
		# 1.3) Se Resuelve M@qpp[0] = P[0] - C@qp[0]- K@q[0], M = I --->qpp0
		qpp = np.zeros((n,m))
		qpp[:,0:1] = P0 - C@qp[:,0:1] - K@q[:,0:1]
		# 1.4) Δt = dt
		# 1.5)
		a1 = M/(β*Δt**2) + γ*C/(β*Δt)
		a2 = M/(β*Δt) + (γ/β - 1)*C
		a3 = (1/(2*β) - 1)*M + Δt*(γ/(2*β) - 1)*C
		# 1.6)
		Kp = K + a1
		# 2.0)
		for i in range(m-1):
			# 2.1) P[i+1] = Φ.T@p[i+1] + a1*q[i] + a2*qp[i] + a3*qpp[i]
			Ppi_1 = Φ.T@p[:,i+1:i+2] + a1@q[:,i:i+1] + a2@qp[:,i:i+1] + a3@qpp[:,i:i+1]
			# 2.2) Se resuelve Kp@q[i+1] = Pp[i+1] --> q[i+1]
			for j in range(n):
				q[:,i+1:i+2][j] =  Ppi_1[j][0]/Kp[j][j]
			# 2.3) qp[i+1] = (γ/(β*Δt))*(q[i+1] - q[i]) + (1 - γ/β)*qp[i] + Δt*(1 - γ/(2*β))*qpp[i]
			qp[:,i+1:i+2] =  (γ/(β*Δt))*(q[:,i+1:i+2] - q[:,i:i+1]) + (1 - γ/β)*qp[:,i:i+1] + Δt*(1 - γ/(2*β))*qpp[:,i:i+1]
			# 2.4) qpp[i+1] = (q[i+1] - q[i])/(β*self.Δt**2) - qp[i]/(β*Δt) - ( 1/(2*β) - 1 )*qpp[i]
			qpp[:,i+1:i+2] = (q[:,i+1:i+2] - q[:,i:i+1])/(β*Δt**2) - qp[:,i:i+1]/(β*Δt) - ( 1/(2*β) - 1 )*qpp[:,i:i+1]

		self.u = Φ@q
		self.up = Φ@qp
		self.upp = Φ@qpp

class Jacobi:

	def __init__(self, A, n):
		# número de ciclos
		self.t = len(A[0])
		self.Ak = copy(A)
		self.s = 0
		self.Pk = np.eye(self.t)
		self.produc_Pk = self.Pk

		for i in range(n):
			self.un_ciclo()

		self.Ω = np.eye(self.t)

		for i in range(self.t):
			self.Ω[i][i] =  self.Ak[i][i]**0.5
		self.Φ = self.produc_Pk

	def P(self, Ak, i, j):
		P = np.eye(self.t)
		self.teta = self.θ(Ak[i][i], Ak[j][j], Ak[i][j])
		P[i][i] = cos(self.teta)
		P[j][j] = cos(self.teta)
		P[i][j] = -sin(self.teta)
		P[j][i] = sin(self.teta)

		return P

	def θ(self, aii, ajj, aij):
		if aii != ajj :
			return 0.5*atan( 2*aij/(aii - ajj) )
		else:
			return np.pi/4

	def un_ciclo(self):

		for i in range(self.t-1):
			for j in range(i+1,self.t):

				self.s +=1
				self.Pk = self.P(self.Ak, i, j)
				self.Ak = self.Pk.T@self.Ak@self.Pk

				self.produc_Pk = self.produc_Pk@self.Pk

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
