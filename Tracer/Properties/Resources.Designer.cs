﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.0
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Tracer.Properties {
    using System;
    
    
    /// <summary>
    ///   A strongly-typed resource class, for looking up localized strings, etc.
    /// </summary>
    // This class was auto-generated by the StronglyTypedResourceBuilder
    // class via a tool like ResGen or Visual Studio.
    // To add or remove a member, edit your .ResX file then rerun ResGen
    // with the /str option, or rebuild your VS project.
    [global::System.CodeDom.Compiler.GeneratedCodeAttribute("System.Resources.Tools.StronglyTypedResourceBuilder", "4.0.0.0")]
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
    [global::System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    internal class Resources {
        
        private static global::System.Resources.ResourceManager resourceMan;
        
        private static global::System.Globalization.CultureInfo resourceCulture;
        
        [global::System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal Resources() {
        }
        
        /// <summary>
        ///   Returns the cached ResourceManager instance used by this class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Resources.ResourceManager ResourceManager {
            get {
                if (object.ReferenceEquals(resourceMan, null)) {
                    global::System.Resources.ResourceManager temp = new global::System.Resources.ResourceManager("Tracer.Properties.Resources", typeof(Resources).Assembly);
                    resourceMan = temp;
                }
                return resourceMan;
            }
        }
        
        /// <summary>
        ///   Overrides the current thread's CurrentUICulture property for all
        ///   resource lookups using this strongly typed resource class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Globalization.CultureInfo Culture {
            get {
                return resourceCulture;
            }
            set {
                resourceCulture = value;
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Cancel.
        /// </summary>
        internal static string Line_Cancel {
            get {
                return ResourceManager.GetString("Line_Cancel", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Render.
        /// </summary>
        internal static string Line_Render {
            get {
                return ResourceManager.GetString("Line_Render", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The directory where all rendered images will be stored..
        /// </summary>
        internal static string Settings_ImageFolder_Description {
            get {
                return ResourceManager.GetString("Settings_ImageFolder_Description", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Done.
        /// </summary>
        internal static string Status_Done {
            get {
                return ResourceManager.GetString("Status_Done", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Rendering image.
        /// </summary>
        internal static string Status_Rendering {
            get {
                return ResourceManager.GetString("Status_Rendering", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Drawing image to bitmap.
        /// </summary>
        internal static string Statuses_Drawing {
            get {
                return ResourceManager.GetString("Statuses_Drawing", resourceCulture);
            }
        }
    }
}
