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
        private const float Mul = 1000;

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
            return Rnd.Next( -( int ) Mul, ( int ) Mul ) / Mul;
        }
    }
}
