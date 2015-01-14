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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Menu));
            this.Status = new System.Windows.Forms.StatusStrip();
            this.Status_Progress = new System.Windows.Forms.ToolStripProgressBar();
            this.Status_Label = new System.Windows.Forms.ToolStripStatusLabel();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.RenderImage = new System.Windows.Forms.PictureBox();
            this.tabControl2 = new System.Windows.Forms.TabControl();
            this.Tab_Output = new System.Windows.Forms.TabPage();
            this.Output_Text = new System.Windows.Forms.RichTextBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.Tab_Render = new System.Windows.Forms.TabPage();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label4 = new System.Windows.Forms.Label();
            this.Settings_AreaDivider = new System.Windows.Forms.NumericUpDown();
            this.Settings_Samples = new System.Windows.Forms.NumericUpDown();
            this.Settings_Depth = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.Render_NextArea = new System.Windows.Forms.Button();
            this.Button_Render = new System.Windows.Forms.Button();
            this.Tab_Scene = new System.Windows.Forms.TabPage();
            this.SceneProperties = new System.Windows.Forms.PropertyGrid();
            this.Tab_Settings = new System.Windows.Forms.TabPage();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.Settings_Devices = new System.Windows.Forms.ComboBox();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.toolStripDropDownButton3 = new System.Windows.Forms.ToolStripDropDownButton();
            this.Scene_SaveButton = new System.Windows.Forms.ToolStripMenuItem();
            this.Scene_LoadButton = new System.Windows.Forms.ToolStripMenuItem();
            this.Scene_LoadDefault = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripDropDownButton2 = new System.Windows.Forms.ToolStripDropDownButton();
            this.Image_Save = new System.Windows.Forms.ToolStripMenuItem();
            this.Notifier = new System.Windows.Forms.NotifyIcon(this.components);
            this.label2 = new System.Windows.Forms.Label();
            this.Settings_SamplesPerRender = new System.Windows.Forms.NumericUpDown();
            this.Status.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).BeginInit();
            this.tabControl2.SuspendLayout();
            this.Tab_Output.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.Tab_Render.SuspendLayout();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_AreaDivider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Samples)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Depth)).BeginInit();
            this.Tab_Scene.SuspendLayout();
            this.Tab_Settings.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_SamplesPerRender)).BeginInit();
            this.SuspendLayout();
            // 
            // Status
            // 
            this.Status.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.Status.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Status_Progress,
            this.Status_Label});
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
            // splitContainer1
            // 
            this.splitContainer1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.splitContainer1.Location = new System.Drawing.Point(0, 28);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.splitContainer2);
            this.splitContainer1.Panel1MinSize = 300;
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.tabControl1);
            this.splitContainer1.Panel2MinSize = 150;
            this.splitContainer1.Size = new System.Drawing.Size(1079, 426);
            this.splitContainer1.SplitterDistance = 839;
            this.splitContainer1.TabIndex = 1;
            // 
            // splitContainer2
            // 
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(0, 0);
            this.splitContainer2.Name = "splitContainer2";
            this.splitContainer2.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.RenderImage);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.tabControl2);
            this.splitContainer2.Size = new System.Drawing.Size(839, 426);
            this.splitContainer2.SplitterDistance = 280;
            this.splitContainer2.TabIndex = 1;
            // 
            // RenderImage
            // 
            this.RenderImage.Cursor = System.Windows.Forms.Cursors.Cross;
            this.RenderImage.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderImage.Location = new System.Drawing.Point(0, 0);
            this.RenderImage.Name = "RenderImage";
            this.RenderImage.Size = new System.Drawing.Size(839, 280);
            this.RenderImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.RenderImage.TabIndex = 0;
            this.RenderImage.TabStop = false;
            // 
            // tabControl2
            // 
            this.tabControl2.Alignment = System.Windows.Forms.TabAlignment.Bottom;
            this.tabControl2.Controls.Add(this.Tab_Output);
            this.tabControl2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl2.Location = new System.Drawing.Point(0, 0);
            this.tabControl2.Name = "tabControl2";
            this.tabControl2.SelectedIndex = 0;
            this.tabControl2.Size = new System.Drawing.Size(839, 142);
            this.tabControl2.TabIndex = 0;
            // 
            // Tab_Output
            // 
            this.Tab_Output.Controls.Add(this.Output_Text);
            this.Tab_Output.Location = new System.Drawing.Point(4, 4);
            this.Tab_Output.Name = "Tab_Output";
            this.Tab_Output.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Output.Size = new System.Drawing.Size(831, 113);
            this.Tab_Output.TabIndex = 0;
            this.Tab_Output.Text = "Output";
            this.Tab_Output.UseVisualStyleBackColor = true;
            // 
            // Output_Text
            // 
            this.Output_Text.Dock = System.Windows.Forms.DockStyle.Fill;
            this.Output_Text.Location = new System.Drawing.Point(3, 3);
            this.Output_Text.Name = "Output_Text";
            this.Output_Text.Size = new System.Drawing.Size(825, 107);
            this.Output_Text.TabIndex = 0;
            this.Output_Text.Text = "";
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
            this.tabControl1.Size = new System.Drawing.Size(236, 426);
            this.tabControl1.TabIndex = 0;
            // 
            // Tab_Render
            // 
            this.Tab_Render.Controls.Add(this.groupBox2);
            this.Tab_Render.Controls.Add(this.Render_NextArea);
            this.Tab_Render.Controls.Add(this.Button_Render);
            this.Tab_Render.Location = new System.Drawing.Point(4, 25);
            this.Tab_Render.Name = "Tab_Render";
            this.Tab_Render.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Render.Size = new System.Drawing.Size(228, 397);
            this.Tab_Render.TabIndex = 1;
            this.Tab_Render.Text = "Rendering";
            this.Tab_Render.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox2.Controls.Add(this.label4);
            this.groupBox2.Controls.Add(this.Settings_SamplesPerRender);
            this.groupBox2.Controls.Add(this.Settings_AreaDivider);
            this.groupBox2.Controls.Add(this.Settings_Samples);
            this.groupBox2.Controls.Add(this.Settings_Depth);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Location = new System.Drawing.Point(7, 65);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(213, 201);
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
            // Settings_AreaDivider
            // 
            this.Settings_AreaDivider.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_AreaDivider.Location = new System.Drawing.Point(6, 128);
            this.Settings_AreaDivider.Maximum = new decimal(new int[] {
            512,
            0,
            0,
            0});
            this.Settings_AreaDivider.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_AreaDivider.Name = "Settings_AreaDivider";
            this.Settings_AreaDivider.Size = new System.Drawing.Size(201, 22);
            this.Settings_AreaDivider.TabIndex = 6;
            this.Settings_AreaDivider.Value = new decimal(new int[] {
            8,
            0,
            0,
            0});
            this.Settings_AreaDivider.ValueChanged += new System.EventHandler(this.Settings_AreaDivider_ValueChanged);
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
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 108);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(84, 17);
            this.label1.TabIndex = 4;
            this.label1.Text = "Area divider";
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
            // Render_NextArea
            // 
            this.Render_NextArea.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Render_NextArea.Location = new System.Drawing.Point(7, 36);
            this.Render_NextArea.Name = "Render_NextArea";
            this.Render_NextArea.Size = new System.Drawing.Size(213, 23);
            this.Render_NextArea.TabIndex = 0;
            this.Render_NextArea.Text = "Next area";
            this.Render_NextArea.UseVisualStyleBackColor = true;
            this.Render_NextArea.Click += new System.EventHandler(this.Render_NextArea_Click);
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
            this.Tab_Scene.Size = new System.Drawing.Size(228, 397);
            this.Tab_Scene.TabIndex = 3;
            this.Tab_Scene.Text = "Scene";
            this.Tab_Scene.UseVisualStyleBackColor = true;
            // 
            // SceneProperties
            // 
            this.SceneProperties.Dock = System.Windows.Forms.DockStyle.Fill;
            this.SceneProperties.Location = new System.Drawing.Point(0, 0);
            this.SceneProperties.Name = "SceneProperties";
            this.SceneProperties.Size = new System.Drawing.Size(228, 397);
            this.SceneProperties.TabIndex = 0;
            // 
            // Tab_Settings
            // 
            this.Tab_Settings.Controls.Add(this.groupBox1);
            this.Tab_Settings.Location = new System.Drawing.Point(4, 25);
            this.Tab_Settings.Name = "Tab_Settings";
            this.Tab_Settings.Padding = new System.Windows.Forms.Padding(3);
            this.Tab_Settings.Size = new System.Drawing.Size(228, 397);
            this.Tab_Settings.TabIndex = 4;
            this.Tab_Settings.Text = "Settings";
            this.Tab_Settings.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.Settings_Devices);
            this.groupBox1.Location = new System.Drawing.Point(7, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(213, 50);
            this.groupBox1.TabIndex = 1;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Device";
            // 
            // Settings_Devices
            // 
            this.Settings_Devices.FormattingEnabled = true;
            this.Settings_Devices.Location = new System.Drawing.Point(5, 21);
            this.Settings_Devices.Name = "Settings_Devices";
            this.Settings_Devices.Size = new System.Drawing.Size(202, 24);
            this.Settings_Devices.TabIndex = 0;
            this.Settings_Devices.SelectedIndexChanged += new System.EventHandler(this.Settings_Devices_SelectedIndexChanged);
            // 
            // toolStrip1
            // 
            this.toolStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripDropDownButton3,
            this.toolStripDropDownButton2});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(1079, 27);
            this.toolStrip1.TabIndex = 2;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // toolStripDropDownButton3
            // 
            this.toolStripDropDownButton3.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Scene_SaveButton,
            this.Scene_LoadButton,
            this.Scene_LoadDefault});
            this.toolStripDropDownButton3.Image = global::Tracer.Properties.Resources.photo;
            this.toolStripDropDownButton3.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripDropDownButton3.Name = "toolStripDropDownButton3";
            this.toolStripDropDownButton3.Size = new System.Drawing.Size(82, 24);
            this.toolStripDropDownButton3.Text = "Scene";
            // 
            // Scene_SaveButton
            // 
            this.Scene_SaveButton.Name = "Scene_SaveButton";
            this.Scene_SaveButton.Size = new System.Drawing.Size(162, 24);
            this.Scene_SaveButton.Text = "Save scene";
            this.Scene_SaveButton.Click += new System.EventHandler(this.Scene_SaveButton_Click);
            // 
            // Scene_LoadButton
            // 
            this.Scene_LoadButton.Name = "Scene_LoadButton";
            this.Scene_LoadButton.Size = new System.Drawing.Size(162, 24);
            this.Scene_LoadButton.Text = "Load scene";
            this.Scene_LoadButton.Click += new System.EventHandler(this.Scene_LoadButton_Click);
            // 
            // Scene_LoadDefault
            // 
            this.Scene_LoadDefault.Name = "Scene_LoadDefault";
            this.Scene_LoadDefault.Size = new System.Drawing.Size(162, 24);
            this.Scene_LoadDefault.Text = "Load default";
            this.Scene_LoadDefault.Click += new System.EventHandler(this.Scene_LoadDefault_Click);
            // 
            // toolStripDropDownButton2
            // 
            this.toolStripDropDownButton2.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Image_Save});
            this.toolStripDropDownButton2.Image = global::Tracer.Properties.Resources.image;
            this.toolStripDropDownButton2.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripDropDownButton2.Name = "toolStripDropDownButton2";
            this.toolStripDropDownButton2.Size = new System.Drawing.Size(85, 24);
            this.toolStripDropDownButton2.Text = "Image";
            // 
            // Image_Save
            // 
            this.Image_Save.Name = "Image_Save";
            this.Image_Save.Size = new System.Drawing.Size(155, 24);
            this.Image_Save.Text = "Save image";
            this.Image_Save.Click += new System.EventHandler(this.Image_Save_Click);
            // 
            // Notifier
            // 
            this.Notifier.Icon = ((System.Drawing.Icon)(resources.GetObject("Notifier.Icon")));
            this.Notifier.Text = "Tracer";
            this.Notifier.Visible = true;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 153);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(133, 17);
            this.label2.TabIndex = 4;
            this.label2.Text = "Samples per render";
            // 
            // Settings_SamplesPerRender
            // 
            this.Settings_SamplesPerRender.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.Settings_SamplesPerRender.Location = new System.Drawing.Point(6, 173);
            this.Settings_SamplesPerRender.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.Settings_SamplesPerRender.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_SamplesPerRender.Name = "Settings_SamplesPerRender";
            this.Settings_SamplesPerRender.Size = new System.Drawing.Size(201, 22);
            this.Settings_SamplesPerRender.TabIndex = 6;
            this.Settings_SamplesPerRender.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.Settings_SamplesPerRender.ValueChanged += new System.EventHandler(this.Settings_SamplesPerRender_ValueChanged);
            // 
            // Menu
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1079, 480);
            this.Controls.Add(this.toolStrip1);
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
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.RenderImage)).EndInit();
            this.tabControl2.ResumeLayout(false);
            this.Tab_Output.ResumeLayout(false);
            this.tabControl1.ResumeLayout(false);
            this.Tab_Render.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_AreaDivider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Samples)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_Depth)).EndInit();
            this.Tab_Scene.ResumeLayout(false);
            this.Tab_Settings.ResumeLayout(false);
            this.groupBox1.ResumeLayout(false);
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Settings_SamplesPerRender)).EndInit();
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
        private System.Windows.Forms.TabPage Tab_Scene;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        public System.Windows.Forms.NumericUpDown Settings_Samples;
        public System.Windows.Forms.NumericUpDown Settings_Depth;
        public System.Windows.Forms.Button Button_Render;
        public System.Windows.Forms.PropertyGrid SceneProperties;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TabPage Tab_Settings;
        private System.Windows.Forms.GroupBox groupBox1;
        public System.Windows.Forms.ComboBox Settings_Devices;
        private System.Windows.Forms.SplitContainer splitContainer2;
        private System.Windows.Forms.TabControl tabControl2;
        private System.Windows.Forms.TabPage Tab_Output;
        private System.Windows.Forms.ToolStrip toolStrip1;
        public System.Windows.Forms.RichTextBox Output_Text;
        private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButton3;
        private System.Windows.Forms.ToolStripMenuItem Scene_SaveButton;
        private System.Windows.Forms.ToolStripMenuItem Scene_LoadButton;
        private System.Windows.Forms.ToolStripMenuItem Scene_LoadDefault;
        private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButton2;
        private System.Windows.Forms.ToolStripMenuItem Image_Save;
        public System.Windows.Forms.NotifyIcon Notifier;
        public System.Windows.Forms.NumericUpDown Settings_AreaDivider;
        private System.Windows.Forms.Label label1;
        public System.Windows.Forms.Button Render_NextArea;
        public System.Windows.Forms.NumericUpDown Settings_SamplesPerRender;
        private System.Windows.Forms.Label label2;

    }
}

