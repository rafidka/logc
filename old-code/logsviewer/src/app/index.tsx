import m from "mithril";
import { MithrilTsxComponent } from "mithril-tsx-component";
import "./style.scss";
import * as data from "../data.json";

const logData: LogData = data;

class LogsClusterWidget extends MithrilTsxComponent<LogCluster> {
  otherLogsVisible = false;

  showHideOtherLogs() {
    this.otherLogsVisible = !this.otherLogsVisible;
  }

  view({ attrs: { logs } }) {
    const firstLog = logs[0];
    const otherLogs = logs.slice(1);

    return (
      <div class="logs-cluster">
        {
          otherLogs.length > 0 ? <button class="expand-logs-btn" onclick={() => this.showHideOtherLogs()}>
            {this.otherLogsVisible ? 'Hide' : 'Show'} {otherLogs.length} similar {otherLogs.length > 1 ? 'logs' : 'log'}
          </button> : ''
        }

        {firstLog}

        {
          (this.otherLogsVisible ? otherLogs : []).map(log => (
            <div class="log">{log}</div>
          ))
        }
      </div >
    );
  }
}

interface LogCluster {
  logs: string[];
}

interface LogData {
  logClusters: LogCluster[];
}

export default class App extends MithrilTsxComponent<{
  logClusters: LogsClusterWidget[]
}> {
  filter = "";

  oninit() {
  }

  updateFilter(value: string) {
    this.filter = value;
  }

  view() {
    const logClusters = logData.logClusters.filter(lc => lc.logs[0].includes(this.filter));
    return (
      <div class="app">
        <input type="text" class="filter-results-input" placeholder="Filter results..." oninput={(e) => { this.updateFilter(e.target.value); }} />
        {logClusters.map(logCluster => (<LogsClusterWidget logs={logCluster.logs} />))}
      </div >
    )
  }

  onbeforeupdate() {
  }
}
