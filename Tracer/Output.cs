
using System;
using System.Linq;

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
                Menu.Output_Text.AppendText( "[" + DateTime.Now.ToLongTimeString( ) + "] " + Line + "\n" );

                if ( Menu.Output_Text.Lines.Length > MaxLines )
                    Menu.Output_Text.Lines =
                        Menu.Output_Text.Lines.Skip( Menu.Output_Text.Lines.Length - MaxLines ).ToArray( );

                Menu.Output_Text.SelectionStart = Menu.Output_Text.Text.Length;
                Menu.Output_Text.ScrollToCaret( );
            } );
        }
    }
}
