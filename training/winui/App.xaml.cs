using Microsoft.UI.Xaml;

namespace CausalVC.Training;

public partial class App : Application
{
    MainWindow? _window;

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, args) => WriteCrash(args.Exception);
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            _window = new MainWindow();
            _window.Activate();
        }
        catch (Exception error)
        {
            // A window that fails while it is built closes without a word: leave the reason behind.
            WriteCrash(error);
            throw;
        }
    }

    /// crash.log next to the program, so a window that closes at once can be reported.
    static void WriteCrash(Exception error)
    {
        try
        {
            File.AppendAllText(Path.Combine(AppContext.BaseDirectory, "crash.log"),
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}\n{error}\n\n");
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}
