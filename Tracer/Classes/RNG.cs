using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tracer.Classes
{
    public class RNG
    {
        private static Random Rnd;

        static RNG( )
        {
            Rnd = new Random( );
        }

        /// <summary>
        /// Returns a float between -1f and 1f.
        /// </summary>
        /// <returns></returns>
        public static float GetUnitFloat( )
        {
            return ( float )( Rnd.NextDouble( ) * 2.0 - 1.0 );
        }

        /// <summary>
        /// Returns a float between 0f and 1f.
        /// </summary>
        /// <returns></returns>
        public static float GetPositiveUnitFloat( )
        {
            return ( float ) Rnd.NextDouble( );
        }
    }
}
