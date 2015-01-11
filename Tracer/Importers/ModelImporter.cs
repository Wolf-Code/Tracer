
using System;
using System.Collections.Generic;
using System.Linq;
using Tracer.Interfaces;

namespace Tracer.Importers
{
    public static class ModelImporter
    {
        private static Dictionary<string, Type> ModelLoaders = new Dictionary<string, Type>
        {
            { "obj", typeof ( OBJImporter ) }
        };

        public static IModel Load( string Path )
        {
            string Extension = Path.Split( '.' ).Last( ).ToLower( );
            if ( !ModelLoaders.ContainsKey( Extension ) )
                throw new Exception( "Unsupported model format: '" + Extension + "'" );

            IModelImporter Importer = Activator.CreateInstance( ModelLoaders[ Extension ] ) as IModelImporter;

            return Importer.Import( Path );
        }
    }
}
