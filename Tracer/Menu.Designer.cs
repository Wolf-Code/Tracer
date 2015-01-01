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
            System.Windows.Forms.TreeNode treeNode1 = new System.Windows.Forms.TreeNode("Objects");
            this.Status = new System.Windows.Forms.StatusStrip();
            this.Status_Progress = new System.Windows.Forms.ToolStripProgressBar();
            this.Status_Label = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripDropDownButton1 = new System.Windows.Forms.ToolStripDropDownButton();
            this.ToolStrip_Button_Save = new System.Windows.Forms.ToolStripMenuItem();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.RenderImage = new System.Windows.Forms.PictureBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.Tab_Render = new System.Windows.Forms.TabPage();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.Settings_Resolution_Width = new System.Windows.Forms.NumericUpDown();
            this.Settings_Resolution_Height = new System.Windows.Forms.NumericUpDown();
            this.Settings_Samples = new System.Windows.Forms.NumericUpDown();
            this.Settings_Depth = new System.Windows.Forms.NumericUpDown();
            this.label5 = new System.Windows.Forms.Label();
            this.Settings_FOV = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.Button_Render = new System.Windows.Forms.Button();
            this.Tab_Scene = new System.Windows.Forms.TabPage();
            this.SceneSplitter = new System.Windows.Forms.SplitContainer();
            this.SceneTree = new System.Windows.Forms.TreeView();
            this.SceneProperties = new System.Windows.Forms.PropertyGrid();
            this.Tab_Settings = new System.Windows.Forms.TabPage();
            this.Settings_BrowseImageFolder = new System.Windows.Forms.LinkLabel();
            this.Settings_ImageFolder = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.Status.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.Tab_Render.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Width)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Height)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Samples)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Depth)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_FOV)).BeginInit();
            this.Tab_Scene.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.SceneSplitter)).BeginInit();
            this.SceneSplitter.Panel1.SuspendLayout();
            this.SceneSplitter.Panel2.SuspendLayout();
            this.SceneSplitter.SuspendLayout();
            this.Tab_Settings.SuspendLayout();
            this.SuspendLayout();
            // 
            // Status
            // 
            this.Status.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.Status.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Status_Progress,
            this.Status_Label,
            this.toolStripDropDownButton1});
            this.Status.Location = new System.Drawing.Point(0, 430);
            this.Status.Name = "Status";
            this.Status.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.Status.Size = new System.Drawing.Size(740, 26);
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
            this.ToolStrip_Button_Save.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.S)));
            this.ToolStrip_Button_Save.Size = new System.Drawing.Size(159, 24);
            this.ToolStrip_Button_Save.Text = "Save";
            this.ToolStrip_Button_Save.Click += new System.EventHandler(this.ToolStrip_Button_Save_Click);
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
            this.splitContainer1.Size = new System.Drawing.Size(740, 430);
            this.splitContainer1.SplitterDistance = 576;
            this.splitContainer1.TabIndex = 1;
            // 
            // RenderImage
            // 
            this.RenderImage.Cursor = System.Windows.Forms.Cursors.Cross;
            this.RenderImage.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderImage.Location = new System.Drawing.Point(0, 0);
            this.RenderImage.Name = "RenderImage";
            this.RenderImage.Size = new System.Drawing.Size(576, 430);
            this.RenderImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.RenderImage.TabIndex = 0;
            this.RenderImage.TabStop = false;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.Tab_Render);
            this.tabControl1.Controls.Add(this.Tab_Scene);
            this.tabControl1.Controls.Add(this.Tab_Settings);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(160, 430);
            this.tabControl1.TabIndex = 0;
            // 
            // Tab_Render
            // 
            this.Tab_Render.Controls.Add(this.groupBox1);
            this.Tab_Render.Controls.Add(this.Settings_Samples);
            this.Tab_Render.Controls.Add(this.Settings_Depth);
            this.Tab_Render.Controls.Add(this.label5);
            this.Tab_Render.Controls.Add(this.Settings_FOV);
            this.Tab_Render.Controls.Add(this.label4);
            this.Tab_Render.Controls.Add(this.label2);
            this.Tab_Render.Controls.Add(this.Button_Render);
            this.Tab_Render.Location = new System.Drawing.Point(4, 25);
            this.Tab_Render.Name = "Tab_Render";
            this.Tab_Render.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Render.Size = new System.Drawing.Size(152, 401);
            this.Tab_Render.TabIndex = 1;
            this.Tab_Render.Text = "Rendering";
            this.Tab_Render.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.Settings_Resolution_Width);
            this.groupBox1.Controls.Add(this.Settings_Resolution_Height);
            this.groupBox1.Location = new System.Drawing.Point(7, 36);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(137, 53);
            this.groupBox1.TabIndex = 7;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Resolution";
            // 
            // Settings_Resolution_Width
            // 
            this.Settings_Resolution_Width.Location = new System.Drawing.Point(6, 21);
            this.Settings_Resolution_Width.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.Settings_Resolution_Width.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Resolution_Width.Name = "Settings_Resolution_Width";
            this.Settings_Resolution_Width.Size = new System.Drawing.Size(59, 22);
            this.Settings_Resolution_Width.TabIndex = 1;
            this.Settings_Resolution_Width.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Resolution_Width.ValueChanged += new System.EventHandler(this.Settings_Resolution_Width_ValueChanged);
            // 
            // Settings_Resolution_Height
            // 
            this.Settings_Resolution_Height.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_Resolution_Height.Location = new System.Drawing.Point(71, 21);
            this.Settings_Resolution_Height.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.Settings_Resolution_Height.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Resolution_Height.Name = "Settings_Resolution_Height";
            this.Settings_Resolution_Height.Size = new System.Drawing.Size(60, 22);
            this.Settings_Resolution_Height.TabIndex = 2;
            this.Settings_Resolution_Height.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Resolution_Height.ValueChanged += new System.EventHandler(this.Settings_Resolution_Height_ValueChanged);
            // 
            // Settings_Samples
            // 
            this.Settings_Samples.Location = new System.Drawing.Point(7, 207);
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
            this.Settings_Samples.Size = new System.Drawing.Size(137, 22);
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
            this.Settings_Depth.Location = new System.Drawing.Point(7, 162);
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
            this.Settings_Depth.Size = new System.Drawing.Size(137, 22);
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
            this.label5.Location = new System.Drawing.Point(10, 187);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(62, 17);
            this.label5.TabIndex = 4;
            this.label5.Text = "Samples";
            // 
            // Settings_FOV
            // 
            this.Settings_FOV.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_FOV.Location = new System.Drawing.Point(7, 117);
            this.Settings_FOV.Maximum = new decimal(new int[] {
            179,
            0,
            0,
            0});
            this.Settings_FOV.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_FOV.Name = "Settings_FOV";
            this.Settings_FOV.Size = new System.Drawing.Size(137, 22);
            this.Settings_FOV.TabIndex = 5;
            this.Settings_FOV.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_FOV.ValueChanged += new System.EventHandler(this.Settings_FOV_ValueChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(10, 142);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(46, 17);
            this.label4.TabIndex = 4;
            this.label4.Text = "Depth";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(10, 96);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(85, 17);
            this.label2.TabIndex = 4;
            this.label2.Text = "Field of view";
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
            // Tab_Scene
            // 
            this.Tab_Scene.Controls.Add(this.SceneSplitter);
            this.Tab_Scene.Location = new System.Drawing.Point(4, 25);
            this.Tab_Scene.Name = "Tab_Scene";
            this.Tab_Scene.Size = new System.Drawing.Size(152, 401);
            this.Tab_Scene.TabIndex = 3;
            this.Tab_Scene.Text = "Scene";
            this.Tab_Scene.UseVisualStyleBackColor = true;
            // 
            // SceneSplitter
            // 
            this.SceneSplitter.Dock = System.Windows.Forms.DockStyle.Fill;
            this.SceneSplitter.Location = new System.Drawing.Point(0, 0);
            this.SceneSplitter.Name = "SceneSplitter";
            this.SceneSplitter.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // SceneSplitter.Panel1
            // 
            this.SceneSplitter.Panel1.Controls.Add(this.SceneTree);
            this.SceneSplitter.Panel1MinSize = 100;
            // 
            // SceneSplitter.Panel2
            // 
            this.SceneSplitter.Panel2.Controls.Add(this.SceneProperties);
            this.SceneSplitter.Panel2MinSize = 100;
            this.SceneSplitter.Size = new System.Drawing.Size(152, 401);
            this.SceneSplitter.SplitterDistance = 177;
            this.SceneSplitter.TabIndex = 0;
            // 
            // SceneTree
            // 
            this.SceneTree.Dock = System.Windows.Forms.DockStyle.Fill;
            this.SceneTree.Location = new System.Drawing.Point(0, 0);
            this.SceneTree.Name = "SceneTree";
            treeNode1.Name = "Objects";
            treeNode1.Text = "Objects";
            this.SceneTree.Nodes.AddRange(new System.Windows.Forms.TreeNode[] {
            treeNode1});
            this.SceneTree.Size = new System.Drawing.Size(152, 177);
            this.SceneTree.TabIndex = 0;
            // 
            // SceneProperties
            // 
            this.SceneProperties.Dock = System.Windows.Forms.DockStyle.Fill;
            this.SceneProperties.Location = new System.Drawing.Point(0, 0);
            this.SceneProperties.Name = "SceneProperties";
            this.SceneProperties.Size = new System.Drawing.Size(152, 220);
            this.SceneProperties.TabIndex = 0;
            // 
            // Tab_Settings
            // 
            this.Tab_Settings.Controls.Add(this.Settings_BrowseImageFolder);
            this.Tab_Settings.Controls.Add(this.Settings_ImageFolder);
            this.Tab_Settings.Controls.Add(this.label3);
            this.Tab_Settings.Location = new System.Drawing.Point(4, 25);
            this.Tab_Settings.Name = "Tab_Settings";
            this.Tab_Settings.Size = new System.Drawing.Size(152, 401);
            this.Tab_Settings.TabIndex = 2;
            this.Tab_Settings.Text = "Settings";
            this.Tab_Settings.UseVisualStyleBackColor = true;
            // 
            // Settings_BrowseImageFolder
            // 
            this.Settings_BrowseImageFolder.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_BrowseImageFolder.AutoSize = true;
            this.Settings_BrowseImageFolder.LinkBehavior = System.Windows.Forms.LinkBehavior.NeverUnderline;
            this.Settings_BrowseImageFolder.Location = new System.Drawing.Point(89, 25);
            this.Settings_BrowseImageFolder.Name = "Settings_BrowseImageFolder";
            this.Settings_BrowseImageFolder.Size = new System.Drawing.Size(54, 17);
            this.Settings_BrowseImageFolder.TabIndex = 3;
            this.Settings_BrowseImageFolder.TabStop = true;
            this.Settings_BrowseImageFolder.Text = "Browse";
            this.Settings_BrowseImageFolder.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.Settings_BrowseImageFolder_LinkClicked);
            // 
            // Settings_ImageFolder
            // 
            this.Settings_ImageFolder.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_ImageFolder.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.Settings_ImageFolder.Location = new System.Drawing.Point(7, 25);
            this.Settings_ImageFolder.Name = "Settings_ImageFolder";
            this.Settings_ImageFolder.ReadOnly = true;
            this.Settings_ImageFolder.Size = new System.Drawing.Size(76, 22);
            this.Settings_ImageFolder.TabIndex = 1;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(4, 4);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(86, 17);
            this.label3.TabIndex = 0;
            this.label3.Text = "Image folder";
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
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.Tab_Render.ResumeLayout(false);
            this.Tab_Render.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Width)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Height)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Samples)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Depth)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_FOV)).EndInit();
            this.Tab_Scene.ResumeLayout(false);
            this.SceneSplitter.Panel1.ResumeLayout(false);
            this.SceneSplitter.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.SceneSplitter)).EndInit();
            this.SceneSplitter.ResumeLayout(false);
            this.Tab_Settings.ResumeLayout(false);
            this.Tab_Settings.PerformLayout();
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
        public System.Windows.Forms.NumericUpDown Settings_Resolution_Height;
        public System.Windows.Forms.NumericUpDown Settings_Resolution_Width;
        public System.Windows.Forms.PictureBox RenderImage;
        private System.Windows.Forms.Label label2;
        public System.Windows.Forms.NumericUpDown Settings_FOV;
        private System.Windows.Forms.TabPage Tab_Settings;
        private System.Windows.Forms.Label label3;
        public System.Windows.Forms.TextBox Settings_ImageFolder;
        private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButton1;
        private System.Windows.Forms.ToolStripMenuItem ToolStrip_Button_Save;
        private System.Windows.Forms.TabPage Tab_Scene;
        private System.Windows.Forms.SplitContainer SceneSplitter;
        private System.Windows.Forms.TreeView SceneTree;
        private System.Windows.Forms.PropertyGrid SceneProperties;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        public System.Windows.Forms.NumericUpDown Settings_Samples;
        public System.Windows.Forms.NumericUpDown Settings_Depth;
        public System.Windows.Forms.Button Button_Render;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.LinkLabel Settings_BrowseImageFolder;

    }
}

