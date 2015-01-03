
using System;

namespace Tracer
{
    public static class Output
    {
        public static int MaxLines = 100;
        private static Menu Menu;
        public static void Initialize( Menu M )
        {
            Menu = M;
        }

        public static void WriteLine( string Format, params object [ ] Data )
        {
            WriteLine( string.Format( Format, Data ) );
        }

        public static void WriteLine( string Line )
        {
            Menu.Perform( ( ) =>
            {
                Menu.Output_Text.AppendText( "[" + DateTime.Now.ToLongTimeString(  ) + "] " + Line + "\n" ); 
                Menu.Output_Text.ScrollToCaret(  );

                if ( Menu.Output_Text.Lines.Length <= MaxLines ) return;

                string [ ] Temp = new string[ MaxLines ];
                Array.Copy( Menu.Output_Text.Lines, Menu.Output_Text.Lines.Length - MaxLines, Temp, 0, MaxLines );

                Menu.Output_Text.Lines = Temp;
            } );
        }
    }
}
