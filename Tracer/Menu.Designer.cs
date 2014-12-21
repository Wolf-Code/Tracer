namespace Tracer
{
    partial class Menu
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose( bool disposing )
        {
            if ( disposing && ( components != null ) )
            {
                components.Dispose( );
            }
            base.Dispose( disposing );
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent( )
        {
            this.Status = new System.Windows.Forms.StatusStrip();
            this.Status_Progress = new System.Windows.Forms.ToolStripProgressBar();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.Tab_Render = new System.Windows.Forms.TabPage();
            this.Button_Render = new System.Windows.Forms.Button();
            this.Status_Label = new System.Windows.Forms.ToolStripStatusLabel();
            this.Status.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.Tab_Render.SuspendLayout();
            this.SuspendLayout();
            // 
            // Status
            // 
            this.Status.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.Status.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Status_Progress,
            this.Status_Label});
            this.Status.Location = new System.Drawing.Point(0, 432);
            this.Status.Name = "Status";
            this.Status.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.Status.Size = new System.Drawing.Size(740, 24);
            this.Status.TabIndex = 0;
            this.Status.Text = "statusStrip1";
            // 
            // Status_Progress
            // 
            this.Status_Progress.Name = "Status_Progress";
            this.Status_Progress.Size = new System.Drawing.Size(100, 18);
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.pictureBox1);
            this.splitContainer1.Panel1MinSize = 300;
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.tabControl1);
            this.splitContainer1.Panel2MinSize = 150;
            this.splitContainer1.Size = new System.Drawing.Size(740, 432);
            this.splitContainer1.SplitterDistance = 576;
            this.splitContainer1.TabIndex = 1;
            // 
            // pictureBox1
            // 
            this.pictureBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox1.Location = new System.Drawing.Point(0, 0);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(576, 432);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.Tab_Render);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(160, 432);
            this.tabControl1.TabIndex = 0;
            // 
            // Tab_Render
            // 
            this.Tab_Render.Controls.Add(this.Button_Render);
            this.Tab_Render.Location = new System.Drawing.Point(4, 25);
            this.Tab_Render.Name = "Tab_Render";
            this.Tab_Render.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Render.Size = new System.Drawing.Size(152, 403);
            this.Tab_Render.TabIndex = 1;
            this.Tab_Render.Text = "Rendering";
            this.Tab_Render.UseVisualStyleBackColor = true;
            // 
            // Button_Render
            // 
            this.Button_Render.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Button_Render.Location = new System.Drawing.Point(7, 7);
            this.Button_Render.Name = "Button_Render";
            this.Button_Render.Size = new System.Drawing.Size(137, 23);
            this.Button_Render.TabIndex = 0;
            this.Button_Render.Text = "Render";
            this.Button_Render.UseVisualStyleBackColor = true;
            this.Button_Render.Click += new System.EventHandler(this.Button_Render_Click);
            // 
            // Status_Label
            // 
            this.Status_Label.Name = "Status_Label";
            this.Status_Label.Size = new System.Drawing.Size(0, 19);
            // 
            // Menu
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(740, 456);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.Status);
            this.Name = "Menu";
            this.Text = "Tracer";
            this.Status.ResumeLayout(false);
            this.Status.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel1.PerformLayout();
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.Tab_Render.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.StatusStrip Status;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage Tab_Render;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button Button_Render;
        private System.Windows.Forms.ToolStripProgressBar Status_Progress;
        private System.Windows.Forms.ToolStripStatusLabel Status_Label;

    }
}

