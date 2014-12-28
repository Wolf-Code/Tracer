using System;
using System.Threading;

namespace Tracer.Classes
{
    public class RNG
    {
        // Made after http://stackoverflow.com/questions/19270507/correct-way-to-use-random-in-multithread-application
        static int Seed = Environment.TickCount;

        static readonly ThreadLocal<Random> Rnd =
            new ThreadLocal<Random>( ( ) => new Random( Interlocked.Increment( ref Seed ) ) );

        /// <summary>
        /// Returns a float between -1f and 1f.
        /// </summary>
        /// <returns></returns>
        public static float GetUnitFloat( )
        {
            return ( float )( Rnd.Value.NextDouble( ) * 2.0 - 1.0 );
        }

        /// <summary>
        /// Returns a float between 0f and 1f.
        /// </summary>
        /// <returns></returns>
        public static float GetPositiveUnitFloat( )
        {
            return ( float ) Rnd.Value.NextDouble( );
        }
    }
}
