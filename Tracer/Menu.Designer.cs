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
            System.Windows.Forms.TreeNode treeNode2 = new System.Windows.Forms.TreeNode("Lights");
            this.Status = new System.Windows.Forms.StatusStrip();
            this.Status_Progress = new System.Windows.Forms.ToolStripProgressBar();
            this.Status_Label = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripDropDownButton1 = new System.Windows.Forms.ToolStripDropDownButton();
            this.ToolStrip_Button_Save = new System.Windows.Forms.ToolStripMenuItem();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.RenderImage = new System.Windows.Forms.PictureBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.Tab_Render = new System.Windows.Forms.TabPage();
            this.Settings_FOV = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.Settings_Resolution_Height = new System.Windows.Forms.NumericUpDown();
            this.Settings_Resolution_Width = new System.Windows.Forms.NumericUpDown();
            this.Button_Render = new System.Windows.Forms.Button();
            this.Tab_Settings = new System.Windows.Forms.TabPage();
            this.Settings_ImageFolder_Browse = new System.Windows.Forms.Button();
            this.Settings_ImageFolder = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.Tab_Scene = new System.Windows.Forms.TabPage();
            this.SceneSplitter = new System.Windows.Forms.SplitContainer();
            this.SceneTree = new System.Windows.Forms.TreeView();
            this.SceneProperties = new System.Windows.Forms.PropertyGrid();
            this.Status.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.Tab_Render.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_FOV)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Height)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Width)).BeginInit();
            this.Tab_Settings.SuspendLayout();
            this.Tab_Scene.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.SceneSplitter)).BeginInit();
            this.SceneSplitter.Panel1.SuspendLayout();
            this.SceneSplitter.Panel2.SuspendLayout();
            this.SceneSplitter.SuspendLayout();
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
            this.RenderImage.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderImage.Location = new System.Drawing.Point(0, 0);
            this.RenderImage.Name = "RenderImage";
            this.RenderImage.Size = new System.Drawing.Size(576, 430);
            this.RenderImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
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
            this.Tab_Render.Controls.Add(this.Settings_FOV);
            this.Tab_Render.Controls.Add(this.label2);
            this.Tab_Render.Controls.Add(this.label1);
            this.Tab_Render.Controls.Add(this.Settings_Resolution_Height);
            this.Tab_Render.Controls.Add(this.Settings_Resolution_Width);
            this.Tab_Render.Controls.Add(this.Button_Render);
            this.Tab_Render.Location = new System.Drawing.Point(4, 25);
            this.Tab_Render.Name = "Tab_Render";
            this.Tab_Render.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Render.Size = new System.Drawing.Size(152, 401);
            this.Tab_Render.TabIndex = 1;
            this.Tab_Render.Text = "Rendering";
            this.Tab_Render.UseVisualStyleBackColor = true;
            // 
            // Settings_FOV
            // 
            this.Settings_FOV.Location = new System.Drawing.Point(6, 103);
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
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 82);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(85, 17);
            this.label2.TabIndex = 4;
            this.label2.Text = "Field of view";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 33);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(75, 17);
            this.label1.TabIndex = 3;
            this.label1.Text = "Resolution";
            // 
            // Settings_Resolution_Height
            // 
            this.Settings_Resolution_Height.Location = new System.Drawing.Point(83, 53);
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
            // Settings_Resolution_Width
            // 
            this.Settings_Resolution_Width.Location = new System.Drawing.Point(6, 53);
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
            this.Settings_Resolution_Width.Size = new System.Drawing.Size(70, 22);
            this.Settings_Resolution_Width.TabIndex = 1;
            this.Settings_Resolution_Width.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_Resolution_Width.ValueChanged += new System.EventHandler(this.Settings_Resolution_Width_ValueChanged);
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
            // Tab_Settings
            // 
            this.Tab_Settings.Controls.Add(this.Settings_ImageFolder_Browse);
            this.Tab_Settings.Controls.Add(this.Settings_ImageFolder);
            this.Tab_Settings.Controls.Add(this.label3);
            this.Tab_Settings.Location = new System.Drawing.Point(4, 25);
            this.Tab_Settings.Name = "Tab_Settings";
            this.Tab_Settings.Size = new System.Drawing.Size(152, 401);
            this.Tab_Settings.TabIndex = 2;
            this.Tab_Settings.Text = "Settings";
            this.Tab_Settings.UseVisualStyleBackColor = true;
            // 
            // Settings_ImageFolder_Browse
            // 
            this.Settings_ImageFolder_Browse.Location = new System.Drawing.Point(121, 25);
            this.Settings_ImageFolder_Browse.Name = "Settings_ImageFolder_Browse";
            this.Settings_ImageFolder_Browse.Size = new System.Drawing.Size(23, 23);
            this.Settings_ImageFolder_Browse.TabIndex = 2;
            this.Settings_ImageFolder_Browse.UseVisualStyleBackColor = true;
            this.Settings_ImageFolder_Browse.Click += new System.EventHandler(this.Settings_ImageFolder_Browse_Click);
            // 
            // Settings_ImageFolder
            // 
            this.Settings_ImageFolder.Location = new System.Drawing.Point(4, 25);
            this.Settings_ImageFolder.Name = "Settings_ImageFolder";
            this.Settings_ImageFolder.ReadOnly = true;
            this.Settings_ImageFolder.Size = new System.Drawing.Size(111, 22);
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
            treeNode2.Name = "Lights";
            treeNode2.Text = "Lights";
            this.SceneTree.Nodes.AddRange(new System.Windows.Forms.TreeNode[] {
            treeNode1,
            treeNode2});
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
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.Tab_Render.ResumeLayout(false);
            this.Tab_Render.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_FOV)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Height)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Resolution_Width)).EndInit();
            this.Tab_Settings.ResumeLayout(false);
            this.Tab_Settings.PerformLayout();
            this.Tab_Scene.ResumeLayout(false);
            this.SceneSplitter.Panel1.ResumeLayout(false);
            this.SceneSplitter.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.SceneSplitter)).EndInit();
            this.SceneSplitter.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage Tab_Render;
        private System.Windows.Forms.Button Button_Render;
        public System.Windows.Forms.StatusStrip Status;
        public System.Windows.Forms.ToolStripProgressBar Status_Progress;
        public System.Windows.Forms.ToolStripStatusLabel Status_Label;
        public System.Windows.Forms.NumericUpDown Settings_Resolution_Height;
        public System.Windows.Forms.NumericUpDown Settings_Resolution_Width;
        public System.Windows.Forms.PictureBox RenderImage;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        public System.Windows.Forms.NumericUpDown Settings_FOV;
        private System.Windows.Forms.TabPage Tab_Settings;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button Settings_ImageFolder_Browse;
        public System.Windows.Forms.TextBox Settings_ImageFolder;
        private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButton1;
        private System.Windows.Forms.ToolStripMenuItem ToolStrip_Button_Save;
        private System.Windows.Forms.TabPage Tab_Scene;
        private System.Windows.Forms.SplitContainer SceneSplitter;
        private System.Windows.Forms.TreeView SceneTree;
        private System.Windows.Forms.PropertyGrid SceneProperties;

    }
}

