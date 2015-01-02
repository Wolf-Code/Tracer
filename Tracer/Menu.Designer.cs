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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Menu));
            this.Status = new System.Windows.Forms.StatusStrip();
            this.Status_Progress = new System.Windows.Forms.ToolStripProgressBar();
            this.Status_Label = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripDropDownButton1 = new System.Windows.Forms.ToolStripDropDownButton();
            this.ToolStrip_Button_Save = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripDropDownButton2 = new System.Windows.Forms.ToolStripDropDownButton();
            this.Scene_SaveButton = new System.Windows.Forms.ToolStripMenuItem();
            this.Scene_LoadButton = new System.Windows.Forms.ToolStripMenuItem();
            this.Scene_LoadDefault = new System.Windows.Forms.ToolStripMenuItem();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.RenderImage = new System.Windows.Forms.PictureBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.Tab_Render = new System.Windows.Forms.TabPage();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label4 = new System.Windows.Forms.Label();
            this.Settings_Samples = new System.Windows.Forms.NumericUpDown();
            this.Settings_Depth = new System.Windows.Forms.NumericUpDown();
            this.label5 = new System.Windows.Forms.Label();
            this.Button_Render = new System.Windows.Forms.Button();
            this.Tab_Scene = new System.Windows.Forms.TabPage();
            this.SceneProperties = new System.Windows.Forms.PropertyGrid();
            this.Status.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.Tab_Render.SuspendLayout();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Samples)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Depth)).BeginInit();
            this.Tab_Scene.SuspendLayout();
            this.SuspendLayout();
            // 
            // Status
            // 
            this.Status.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.Status.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Status_Progress,
            this.Status_Label,
            this.toolStripDropDownButton1,
            this.toolStripDropDownButton2});
            this.Status.Location = new System.Drawing.Point(0, 454);
            this.Status.Name = "Status";
            this.Status.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.Status.Size = new System.Drawing.Size(1079, 26);
            this.Status.TabIndex = 0;
            this.Status.Text = "statusStrip1";
            // 
            // Status_Progress
            // 
            this.Status_Progress.Name = "Status_Progress";
            this.Status_Progress.Size = new System.Drawing.Size(100, 20);
            // 
            // Status_Label
            // 
            this.Status_Label.Name = "Status_Label";
            this.Status_Label.Size = new System.Drawing.Size(0, 21);
            // 
            // toolStripDropDownButton1
            // 
            this.toolStripDropDownButton1.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripDropDownButton1.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.ToolStrip_Button_Save});
            this.toolStripDropDownButton1.Image = ((System.Drawing.Image)(resources.GetObject("toolStripDropDownButton1.Image")));
            this.toolStripDropDownButton1.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripDropDownButton1.Name = "toolStripDropDownButton1";
            this.toolStripDropDownButton1.Size = new System.Drawing.Size(34, 24);
            this.toolStripDropDownButton1.Text = "toolStripDropDownButton1";
            // 
            // ToolStrip_Button_Save
            // 
            this.ToolStrip_Button_Save.Name = "ToolStrip_Button_Save";
            this.ToolStrip_Button_Save.Size = new System.Drawing.Size(155, 24);
            this.ToolStrip_Button_Save.Text = "Save image";
            this.ToolStrip_Button_Save.Click += new System.EventHandler(this.ToolStrip_Button_Save_Click);
            // 
            // toolStripDropDownButton2
            // 
            this.toolStripDropDownButton2.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripDropDownButton2.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Scene_SaveButton,
            this.Scene_LoadButton,
            this.Scene_LoadDefault});
            this.toolStripDropDownButton2.Image = ((System.Drawing.Image)(resources.GetObject("toolStripDropDownButton2.Image")));
            this.toolStripDropDownButton2.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripDropDownButton2.Name = "toolStripDropDownButton2";
            this.toolStripDropDownButton2.Size = new System.Drawing.Size(62, 24);
            this.toolStripDropDownButton2.Text = "Scene";
            // 
            // Scene_SaveButton
            // 
            this.Scene_SaveButton.Name = "Scene_SaveButton";
            this.Scene_SaveButton.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.S)));
            this.Scene_SaveButton.Size = new System.Drawing.Size(168, 24);
            this.Scene_SaveButton.Text = "Save";
            this.Scene_SaveButton.Click += new System.EventHandler(this.Scene_SaveButton_Click);
            // 
            // Scene_LoadButton
            // 
            this.Scene_LoadButton.Name = "Scene_LoadButton";
            this.Scene_LoadButton.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.O)));
            this.Scene_LoadButton.Size = new System.Drawing.Size(168, 24);
            this.Scene_LoadButton.Text = "Load";
            this.Scene_LoadButton.Click += new System.EventHandler(this.Scene_LoadButton_Click);
            // 
            // Scene_LoadDefault
            // 
            this.Scene_LoadDefault.Name = "Scene_LoadDefault";
            this.Scene_LoadDefault.Size = new System.Drawing.Size(168, 24);
            this.Scene_LoadDefault.Text = "Default scene";
            this.Scene_LoadDefault.Click += new System.EventHandler(this.Scene_LoadDefault_Click);
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.RenderImage);
            this.splitContainer1.Panel1MinSize = 300;
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.tabControl1);
            this.splitContainer1.Panel2MinSize = 150;
            this.splitContainer1.Size = new System.Drawing.Size(1079, 454);
            this.splitContainer1.SplitterDistance = 839;
            this.splitContainer1.TabIndex = 1;
            // 
            // RenderImage
            // 
            this.RenderImage.Cursor = System.Windows.Forms.Cursors.Cross;
            this.RenderImage.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderImage.Location = new System.Drawing.Point(0, 0);
            this.RenderImage.Name = "RenderImage";
            this.RenderImage.Size = new System.Drawing.Size(839, 454);
            this.RenderImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.RenderImage.TabIndex = 0;
            this.RenderImage.TabStop = false;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.Tab_Render);
            this.tabControl1.Controls.Add(this.Tab_Scene);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(236, 454);
            this.tabControl1.TabIndex = 0;
            // 
            // Tab_Render
            // 
            this.Tab_Render.Controls.Add(this.groupBox2);
            this.Tab_Render.Controls.Add(this.Button_Render);
            this.Tab_Render.Location = new System.Drawing.Point(4, 25);
            this.Tab_Render.Name = "Tab_Render";
            this.Tab_Render.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Render.Size = new System.Drawing.Size(228, 425);
            this.Tab_Render.TabIndex = 1;
            this.Tab_Render.Text = "Rendering";
            this.Tab_Render.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox2.Controls.Add(this.label4);
            this.groupBox2.Controls.Add(this.Settings_Samples);
            this.groupBox2.Controls.Add(this.Settings_Depth);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Location = new System.Drawing.Point(7, 36);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(213, 113);
            this.groupBox2.TabIndex = 8;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Settings";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 18);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(46, 17);
            this.label4.TabIndex = 4;
            this.label4.Text = "Depth";
            // 
            // Settings_Samples
            // 
            this.Settings_Samples.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_Samples.Location = new System.Drawing.Point(6, 83);
            this.Settings_Samples.Maximum = new decimal(new int[] {
            50000,
            0,
            0,
            0});
            this.Settings_Samples.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Samples.Name = "Settings_Samples";
            this.Settings_Samples.Size = new System.Drawing.Size(201, 22);
            this.Settings_Samples.TabIndex = 6;
            this.Settings_Samples.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Samples.ValueChanged += new System.EventHandler(this.Settings_Samples_ValueChanged);
            // 
            // Settings_Depth
            // 
            this.Settings_Depth.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_Depth.Enabled = false;
            this.Settings_Depth.Location = new System.Drawing.Point(6, 38);
            this.Settings_Depth.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.Settings_Depth.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Depth.Name = "Settings_Depth";
            this.Settings_Depth.Size = new System.Drawing.Size(201, 22);
            this.Settings_Depth.TabIndex = 6;
            this.Settings_Depth.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Depth.ValueChanged += new System.EventHandler(this.Settings_Depth_ValueChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(6, 63);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(62, 17);
            this.label5.TabIndex = 4;
            this.label5.Text = "Samples";
            // 
            // Button_Render
            // 
            this.Button_Render.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Button_Render.Location = new System.Drawing.Point(7, 7);
            this.Button_Render.Name = "Button_Render";
            this.Button_Render.Size = new System.Drawing.Size(213, 23);
            this.Button_Render.TabIndex = 0;
            this.Button_Render.Text = "Render";
            this.Button_Render.UseVisualStyleBackColor = true;
            this.Button_Render.Click += new System.EventHandler(this.Button_Render_Click);
            // 
            // Tab_Scene
            // 
            this.Tab_Scene.Controls.Add(this.SceneProperties);
            this.Tab_Scene.Location = new System.Drawing.Point(4, 25);
            this.Tab_Scene.Name = "Tab_Scene";
            this.Tab_Scene.Size = new System.Drawing.Size(228, 425);
            this.Tab_Scene.TabIndex = 3;
            this.Tab_Scene.Text = "Scene";
            this.Tab_Scene.UseVisualStyleBackColor = true;
            // 
            // SceneProperties
            // 
            this.SceneProperties.Dock = System.Windows.Forms.DockStyle.Fill;
            this.SceneProperties.Location = new System.Drawing.Point(0, 0);
            this.SceneProperties.Name = "SceneProperties";
            this.SceneProperties.Size = new System.Drawing.Size(228, 425);
            this.SceneProperties.TabIndex = 0;
            // 
            // Menu
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1079, 480);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.Status);
            this.Name = "Menu";
            this.Text = "Tracer";
            this.Status.ResumeLayout(false);
            this.Status.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.Tab_Render.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Samples)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Depth)).EndInit();
            this.Tab_Scene.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage Tab_Render;
        public System.Windows.Forms.StatusStrip Status;
        public System.Windows.Forms.ToolStripProgressBar Status_Progress;
        public System.Windows.Forms.ToolStripStatusLabel Status_Label;
        public System.Windows.Forms.PictureBox RenderImage;
        private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButton1;
        private System.Windows.Forms.ToolStripMenuItem ToolStrip_Button_Save;
        private System.Windows.Forms.TabPage Tab_Scene;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        public System.Windows.Forms.NumericUpDown Settings_Samples;
        public System.Windows.Forms.NumericUpDown Settings_Depth;
        public System.Windows.Forms.Button Button_Render;
        public System.Windows.Forms.PropertyGrid SceneProperties;
        private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButton2;
        private System.Windows.Forms.ToolStripMenuItem Scene_SaveButton;
        private System.Windows.Forms.ToolStripMenuItem Scene_LoadButton;
        private System.Windows.Forms.ToolStripMenuItem Scene_LoadDefault;
        private System.Windows.Forms.GroupBox groupBox2;

    }
}

