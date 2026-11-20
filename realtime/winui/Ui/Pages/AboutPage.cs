using CausalVC.Realtime.Engine;

namespace CausalVC.Realtime.Ui.Pages;

/// This version and updates (pages.rs: about_page). Shown whether or not the engine runs.
public sealed class AboutPage : PageBase
{
    public AboutPage(IHost host, SettingsSection updates) : base(host)
    {
        Column.Children.Add(updates);
    }

    public override void Update(EngineState state) { }
}
