using System;
using System.ComponentModel.Design;
using System.Linq;
using System.Reflection;
using Tracer.Classes.Objects;

namespace Tracer.Classes
{
    // Kudos to Keith DeGrace at http://www.pcreview.co.uk/forums/propertygrid-collectioneditor-problem-t1316579.html
    // for posting his solution, upon which mine is based.
    public class ControlCollectionEditor : CollectionEditor
    {
        public ControlCollectionEditor( Type type )
            : base( type )
        {
            
        }

        protected override Type [ ] CreateNewItemTypes( )
        {
            Type [ ] newItemTypes =
                Assembly.GetExecutingAssembly( )
                    .GetTypes( )
                    .Where( O => O.IsSubclassOf( typeof ( GraphicsObject ) ) )
                    .ToArray( );

            return newItemTypes;
        }
    }
}
