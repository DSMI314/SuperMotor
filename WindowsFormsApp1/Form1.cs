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

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        private int prevMode = -1;
        public Form1()
        {
            InitializeComponent();
            string path = System.Windows.Forms.Application.StartupPath;

            // Call this procedure when the application starts.
            // Set to 1 second.
            timer1.Interval = 100;

            // Enable timer.
            timer1.Enabled = true;


        }
        
        private void Form1_Load(object sender, EventArgs e)
        {
            buttonMode0.BackColor = Color.Gray;
            buttonMode1.BackColor = Color.Gray;
            buttonMode2.BackColor = Color.Gray;
            buttonMode3.BackColor = Color.Gray;
            ChangeLight(-1);
        }

        private void ChangeLight(int status)
        {
            switch (status)
            {
                case -1:
                    lightPictureBox.Image = Properties.Resources.redLight;
                    warningLabel.ForeColor = Color.Red;
                    warningLabel.Text = "Closed";
                    break;
                case 0:
                    lightPictureBox.Image = Properties.Resources.yellowLight;
                    warningLabel.ForeColor = Color.Yellow;
                    warningLabel.Text = "Alert";
                    break;
                case 1:
                    lightPictureBox.Image = Properties.Resources.greenLight;
                    warningLabel.ForeColor = Color.Green;
                    warningLabel.Text = "Stable";
                    break;
            }
        }
        private void timer1_Tick(object sender, EventArgs e)
        {
            int prediction = prevMode;
            try
            {
                StreamReader sr = new StreamReader(@"prediction.txt");
                while (!sr.EndOfStream)
                {
                    string line = sr.ReadLine();
                    prediction = Convert.ToInt32(line);
                    break;
                }
                sr.Close();
            }
            catch (IOException)
            {
                
            }
            if(prediction >= 0 && prediction <= 3)
            {
                if(prediction != prevMode)
                {
                    ChangeLight(0);
                }
                else
                {
                    ChangeLight(1);
                }
                prevMode = prediction;
            }
            switch (prediction)
            {
                case -1:
                    buttonMode0.BackColor = Color.Gray;
                    buttonMode1.BackColor = Color.Gray;
                    buttonMode2.BackColor = Color.Gray;
                    buttonMode3.BackColor = Color.Gray;
                    break;
                case 0:
                    buttonMode0.BackColor = Color.Tomato;
                    buttonMode1.BackColor = Color.Gray;
                    buttonMode2.BackColor = Color.Gray;
                    buttonMode3.BackColor = Color.Gray;
                    break;
                case 1:
                    buttonMode0.BackColor = Color.Gray;
                    buttonMode1.BackColor = Color.Tomato;
                    buttonMode2.BackColor = Color.Gray;
                    buttonMode3.BackColor = Color.Gray;
                    break;
                case 2:
                    buttonMode0.BackColor = Color.Gray;
                    buttonMode1.BackColor = Color.Gray;
                    buttonMode2.BackColor = Color.Tomato;
                    buttonMode3.BackColor = Color.Gray;
                    break;
                case 3:
                    buttonMode0.BackColor = Color.Gray;
                    buttonMode1.BackColor = Color.Gray;
                    buttonMode2.BackColor = Color.Gray;
                    buttonMode3.BackColor = Color.Tomato;
                    break;
            }
        }

        private void warningLabel_Click(object sender, EventArgs e)
        {

        }
    }
}
