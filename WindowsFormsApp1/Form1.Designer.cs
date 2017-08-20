namespace WindowsFormsApp1
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this._titleLabel = new System.Windows.Forms.Label();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this._BTlight01PictureBox = new System.Windows.Forms.PictureBox();
            this._BT01Label = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            ((System.ComponentModel.ISupportInitialize)(this._BTlight01PictureBox)).BeginInit();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // _titleLabel
            // 
            this._titleLabel.AutoSize = true;
            this._titleLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 60F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this._titleLabel.Location = new System.Drawing.Point(58, 61);
            this._titleLabel.Name = "_titleLabel";
            this._titleLabel.Size = new System.Drawing.Size(305, 91);
            this._titleLabel.TabIndex = 0;
            this._titleLabel.Text = "Current";
            // 
            // timer1
            // 
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // _BTlight01PictureBox
            // 
            this._BTlight01PictureBox.Location = new System.Drawing.Point(3, 4);
            this._BTlight01PictureBox.Name = "_BTlight01PictureBox";
            this._BTlight01PictureBox.Size = new System.Drawing.Size(80, 80);
            this._BTlight01PictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this._BTlight01PictureBox.TabIndex = 8;
            this._BTlight01PictureBox.TabStop = false;
            // 
            // _BT01Label
            // 
            this._BT01Label.AutoSize = true;
            this._BT01Label.Font = new System.Drawing.Font("Microsoft Sans Serif", 30F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this._BT01Label.Location = new System.Drawing.Point(89, 24);
            this._BT01Label.Name = "_BT01Label";
            this._BT01Label.Size = new System.Drawing.Size(188, 46);
            this._BT01Label.TabIndex = 9;
            this._BT01Label.Text = "BT Name";
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this._BTlight01PictureBox);
            this.panel1.Controls.Add(this._BT01Label);
            this.panel1.Location = new System.Drawing.Point(41, 182);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(332, 89);
            this.panel1.TabIndex = 10;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(425, 311);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this._titleLabel);
            this.Name = "Form1";
            this.Text = "Sensor BT Demo";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this._BTlight01PictureBox)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label _titleLabel;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.PictureBox _BTlight01PictureBox;
        private System.Windows.Forms.Label _BT01Label;
        private System.Windows.Forms.Panel panel1;
    }
}

