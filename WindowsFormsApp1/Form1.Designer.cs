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
            this.buttonMode1 = new System.Windows.Forms.Button();
            this.buttonMode2 = new System.Windows.Forms.Button();
            this.buttonMode3 = new System.Windows.Forms.Button();
            this.buttonMode0 = new System.Windows.Forms.Button();
            this.warningLabel = new System.Windows.Forms.Label();
            this.lightPictureBox = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.lightPictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // _titleLabel
            // 
            this._titleLabel.AutoSize = true;
            this._titleLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 50F);
            this._titleLabel.Location = new System.Drawing.Point(140, 72);
            this._titleLabel.Name = "_titleLabel";
            this._titleLabel.Size = new System.Drawing.Size(255, 76);
            this._titleLabel.TabIndex = 0;
            this._titleLabel.Text = "Current";
            // 
            // timer1
            // 
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // buttonMode1
            // 
            this.buttonMode1.BackColor = System.Drawing.SystemColors.Control;
            this.buttonMode1.Enabled = false;
            this.buttonMode1.Font = new System.Drawing.Font("Microsoft Sans Serif", 25F);
            this.buttonMode1.ForeColor = System.Drawing.SystemColors.ControlText;
            this.buttonMode1.Location = new System.Drawing.Point(305, 279);
            this.buttonMode1.Name = "buttonMode1";
            this.buttonMode1.Size = new System.Drawing.Size(150, 200);
            this.buttonMode1.TabIndex = 3;
            this.buttonMode1.Text = "Mode 1";
            this.buttonMode1.UseVisualStyleBackColor = false;
            // 
            // buttonMode2
            // 
            this.buttonMode2.BackColor = System.Drawing.SystemColors.Control;
            this.buttonMode2.Enabled = false;
            this.buttonMode2.Font = new System.Drawing.Font("Microsoft Sans Serif", 25F);
            this.buttonMode2.ForeColor = System.Drawing.SystemColors.ControlText;
            this.buttonMode2.Location = new System.Drawing.Point(538, 279);
            this.buttonMode2.Name = "buttonMode2";
            this.buttonMode2.Size = new System.Drawing.Size(150, 200);
            this.buttonMode2.TabIndex = 4;
            this.buttonMode2.Text = "Mode 2";
            this.buttonMode2.UseVisualStyleBackColor = false;
            // 
            // buttonMode3
            // 
            this.buttonMode3.BackColor = System.Drawing.SystemColors.Control;
            this.buttonMode3.Enabled = false;
            this.buttonMode3.Font = new System.Drawing.Font("Microsoft Sans Serif", 25F);
            this.buttonMode3.ForeColor = System.Drawing.SystemColors.ControlText;
            this.buttonMode3.Location = new System.Drawing.Point(773, 279);
            this.buttonMode3.Name = "buttonMode3";
            this.buttonMode3.Size = new System.Drawing.Size(150, 200);
            this.buttonMode3.TabIndex = 5;
            this.buttonMode3.Text = "Mode 3";
            this.buttonMode3.UseVisualStyleBackColor = false;
            // 
            // buttonMode0
            // 
            this.buttonMode0.BackColor = System.Drawing.SystemColors.Control;
            this.buttonMode0.Enabled = false;
            this.buttonMode0.Font = new System.Drawing.Font("Microsoft Sans Serif", 25F);
            this.buttonMode0.ForeColor = System.Drawing.SystemColors.ControlText;
            this.buttonMode0.Location = new System.Drawing.Point(73, 279);
            this.buttonMode0.Name = "buttonMode0";
            this.buttonMode0.Size = new System.Drawing.Size(150, 200);
            this.buttonMode0.TabIndex = 6;
            this.buttonMode0.Text = "Mode 0";
            this.buttonMode0.UseVisualStyleBackColor = false;
            // 
            // warningLabel
            // 
            this.warningLabel.AutoSize = true;
            this.warningLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 40F);
            this.warningLabel.ForeColor = System.Drawing.Color.Red;
            this.warningLabel.Location = new System.Drawing.Point(644, 85);
            this.warningLabel.Name = "warningLabel";
            this.warningLabel.Size = new System.Drawing.Size(350, 63);
            this.warningLabel.TabIndex = 7;
            this.warningLabel.Text = "warningLabel";
            // 
            // lightPictureBox
            // 
            this.lightPictureBox.Location = new System.Drawing.Point(538, 72);
            this.lightPictureBox.Name = "lightPictureBox";
            this.lightPictureBox.Size = new System.Drawing.Size(100, 100);
            this.lightPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.lightPictureBox.TabIndex = 8;
            this.lightPictureBox.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(984, 561);
            this.Controls.Add(this.lightPictureBox);
            this.Controls.Add(this.warningLabel);
            this.Controls.Add(this.buttonMode0);
            this.Controls.Add(this.buttonMode3);
            this.Controls.Add(this.buttonMode2);
            this.Controls.Add(this.buttonMode1);
            this.Controls.Add(this._titleLabel);
            this.Name = "Form1";
            this.Text = "Sensor Project Demo";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.lightPictureBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label _titleLabel;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Button buttonMode1;
        private System.Windows.Forms.Button buttonMode2;
        private System.Windows.Forms.Button buttonMode3;
        private System.Windows.Forms.Button buttonMode0;
        private System.Windows.Forms.Label warningLabel;
        private System.Windows.Forms.PictureBox lightPictureBox;
    }
}

