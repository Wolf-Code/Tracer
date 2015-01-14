using System;
using System.Diagnostics;
namespace Utilib.Diagnostics
{
    public class Timer
    {
        private readonly Stopwatch Watch;
        private TimeSpan TimeSoFar;
        public TimeSpan Average { private set; get; }
        public int Rounds { private set; get; }

        public TimeSpan TotalTime
        {
            get { return TimeSoFar; }
        }

        public Timer( )
        {
            Watch = new Stopwatch( );
            TimeSoFar = new TimeSpan( 0 );
            Average = new TimeSpan( 0 );
            Rounds = 0;
        }

        /// <summary>
        /// Starts a round.
        /// </summary>
        public void StartRound( )
        {
            Watch.Start( );
        }

        public TimeSpan StopRound( )
        {
            Watch.Stop( );
         
            TimeSpan T = Watch.Elapsed;
            TimeSoFar += T;

            Average = Rounds++ == 0 ? T : new TimeSpan( TimeSoFar.Ticks / Rounds );

            Watch.Reset( );

            return T;
        }
    }
}