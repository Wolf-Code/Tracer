using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Tracer
{
    public static class Content
    {
        public static void Save<T>( string Path, T Data ) where T : new( )
        {
            using ( Stream S = File.Open( Path, FileMode.Create ) )
            {
                BinaryFormatter Formatter = new BinaryFormatter( );
                Formatter.Serialize( S, Data );
            }
        }

        public static T Load<T>( string Path ) where T : class
        {
            using ( Stream S = File.Open( Path, FileMode.Open ) )
            {
                BinaryFormatter Formatter = new BinaryFormatter( );
                return Formatter.Deserialize( S ) as T;
            }
        }
    }
}
