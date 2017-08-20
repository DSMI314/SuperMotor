using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Globalization;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        private int prevMode = -1;
        private string prevName = "Unknown";
        public Form1()
        {
            InitializeComponent();
            string path = System.Windows.Forms.Application.StartupPath;

            // Call this procedure when the application starts.
            // Set to 1 second.
            timer1.Interval = 200;

            // Enable timer.
            timer1.Enabled = true;


        }
        
        private void Form1_Load(object sender, EventArgs e)
        {
            ChangeLight("Unknown", -1);
        }

        private void ChangeLight(string bt_name, int status)
        {
            _BT01Label.Text = bt_name;
            switch (status)
            {
                case -1:
                    _BTlight01PictureBox.Image = Properties.Resources.redLight;
                    break;
                case 0:
                    _BTlight01PictureBox.Image = Properties.Resources.greenLight;
                    break;
                case 1:
                    _BTlight01PictureBox.Image = Properties.Resources.yellowLight;
                    break;
            }
        }
        private void timer1_Tick(object sender, EventArgs e)
        {
            int prediction = prevMode;
            string bt_name = prevName;
            try
            {
                StreamReader sr = new StreamReader(@"E:\\python\\Supermotor\\prediction.txt");
                while (!sr.EndOfStream)
                {
                    bt_name = sr.ReadLine();

                    string line = sr.ReadLine();
                    prediction = Convert.ToInt32(line);
                    break;
                }
                sr.Close();
            }
            
            catch (IOException)
            {
                
            }

            ChangeLight(prevName = bt_name, prevMode = prediction);
        }
    }
}
