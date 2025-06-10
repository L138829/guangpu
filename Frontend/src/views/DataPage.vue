<template>
  <div class="dashboard">
    
    <!-- 左侧筛选栏 -->
    <el-aside width="200px" class="sidebar">
      <div class="filter-group">
        <el-input
          v-model="filters.event"
          placeholder="请输入事件名称"
          clearable
          @clear="handleSearch"
          @keyup.enter.native="handleSearch"
        >
          <el-button slot="append" icon="el-icon-search" @click="handleSearch" />
        </el-input>

        <el-input
          v-model="filters.name"
          placeholder="请输入名称"
          clearable
          @clear="handleSearch"
          @keyup.enter.native="handleSearch"
          class="mt-10"
        >
          <el-button slot="append" icon="el-icon-search" @click="handleSearch" />
        </el-input>
      </div>
      <!-- 上传间隔 -->
       <div class="mt-10">
         <el-form-item label="上传间隔 (s)">
           <el-input-number
             v-model="uploadInterval"
             :min="20"
             :max="3600"
             :step="10"
             controls-position="right"
           />
         </el-form-item>
       </div>

      <div class="mt-20">属性</div>
      <el-menu
        :default-active="filters.name"
        @select="onMetricSelect"
        class="metric-menu"
      >
        <el-menu-item
          v-for="m in metricsList"
          :key="m"
          :index="m"
        >
          {{ m }}
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- 右侧主区域 -->
    <el-container>
      <!-- 顶部工具栏 -->
      <el-header class="toolbar">
        <div class="view-switcher">
          <!-- 列表视图按钮，三条横线 -->
          <div
            class="switcher-btn"
            :class="{ active: viewMode==='list' }"
            @click="viewMode='list'"
          >
            <div class="list-icon">
              <span></span><span></span><span></span>
            </div>
          </div>
          <!-- 网格视图按钮，四个方块 -->
          <div
            class="switcher-btn"
            :class="{ active: viewMode==='grid' }"
            @click="viewMode='grid'"
          >
            <div class="grid-icon">
              <div></div><div></div>
              <div></div><div></div>
            </div>
          </div>
        </div>
      </el-header>

      <!-- 内容区 -->
      <el-main class="content">
        <!-- 列表视图 -->
        <el-table
          v-if="viewMode === 'list'"
          :data="pagedData"
          stripe
          border
          fit
          style="width: 100%;"
        >
          <el-table-column prop="name" label="名称" />
          <el-table-column prop="valueWithUnit" label="值" />
          <el-table-column prop="updateTime" label="更新时间" />
          <el-table-column label="操作" width="140">
            <template #default="{ row }">
              <el-button type="text" icon="el-icon-edit" circle />
              <el-button type="text" icon="el-icon-refresh" circle />
              <el-button type="text" icon="el-icon-more" circle />
            </template>
          </el-table-column>
        </el-table>

        <!-- 网格视图 -->
        <div v-else class="card-grid">
          <el-card
            v-for="item in pagedData"
            :key="item.id"
            class="data-card"
          >
            <div class="card-header">
              <span>{{ item.name }}</span>
              <div class="actions">
                <el-button type="text" icon="el-icon-edit" circle />
                <el-button type="text" icon="el-icon-refresh" circle />
                <el-button type="text" icon="el-icon-more" circle />
              </div>
            </div>
            <div class="card-body">
              <div class="value">{{ item.valueWithUnit }}</div>
              <div class="time">{{ item.updateTime }}</div>
            </div>
          </el-card>
        </div>
      </el-main>

      <!-- 底部分页（仅列表模式） -->
      <el-footer v-if="viewMode==='list'" class="footer">
        <el-pagination
          background
          layout="total, prev, pager, next"
          :total="filteredData.length"
          :page-size="listPageSize"
          :current-page.sync="pager.current"
          @current-change="handlePageChange"
        />
      </el-footer>
    </el-container>
  </div>
</template>

<script>
import mqtt from 'mqtt'
import axios from 'axios'

export default {
  name: 'DataDashboard',
  data() {
    return {
      viewMode: 'list',           // 'list' or 'grid'
      filters: { event: '', name: '' },
      // 新增：上传间隔，单位秒，最小20，最大3600
      uploadInterval: 20,
      pager: { current: 1 },
      listPageSize: 8,            // 列表每页固定 8 条
      allData: [],                // 渲染用的数据数组
      metricsList: [
        '溶解氧','总磷','叶绿素','地表水水质','黑臭水体状态','富营养状态',
        '总氮','氨氮','化学需氧量','PH','高锰酸钾指数','浊度'
      ],
      fieldMap: {                 // 后端字段到展示名称/单位映射
        Chla:     { label: '叶绿素',    unit: 'µg/L' },
        TP:       { label: '总磷',      unit: 'mg/L' },
        COD:      { label: '化学需氧量', unit: 'mg/L' },
        NH3N:     { label: '氨氮',      unit: 'mg/L' },
        DO:       { label: '溶解氧',    unit: 'mg/L' },
        PH:       { label: 'PH',        unit: ''      },
        Turbidity:{ label: '浊度',      unit: 'FTU'  },
        Water:         { label: '地表水水质',  unit: '' },
        Trophic_State: { label: '富营养状态',  unit: '' },
        Black_water:   { label: '黑臭水体状态',unit: '' }
      }
    }
  },
  computed: {
    filteredData() {
      return this.allData.filter(item => {
        const ok1 = !this.filters.event || item.name.includes(this.filters.event)
        const ok2 = !this.filters.name  || item.name === this.filters.name
        return ok1 && ok2
      })
    },
    pagedData() {
      const arr = this.filteredData.map(item => ({
        ...item,
        valueWithUnit: `${item.value}${item.unit}`
      }))
      if (this.viewMode === 'list') {
        const start = (this.pager.current - 1) * this.listPageSize
        return arr.slice(start, start + this.listPageSize)
      }
      return arr
    }
  },
  methods: {
    // =========== 列表／筛选／分页 相关 =============
    handleSearch() {
      this.pager.current = 1;
    },
    onMetricSelect(name) {
      this.filters.name = name;
      this.pager.current = 1;
    },
    handlePageChange(page) {
      this.pager.current = page;
    },

    // =========== 上传间隔控制 相关 =============
    /** 启动或重启上传定时器 */
    startUploadTimer() {
      // 如果已有定时器，先清除
      if (this._uploadTimer) {
        clearInterval(this._uploadTimer);
      }
      // 使用当前 uploadInterval（秒）创建新定时任务
      this._uploadTimer = setInterval(() => {
        this.pushData();
      }, this.uploadInterval * 1000);
    },

    /** 真正执行一次数据推送 */
    pushData() {
      // TODO: 根据你的后端接口，发送 this.allData
      // 例如 axios.post('/api/upload', { data: this.allData })
      console.log('推送数据：', this.allData);
    },

    /** 当用户修改上传间隔后调用 */
    updateInterval(newInterval) {
      // （可选）先通知后端
      // axios.post('/api/config', { uploadInterval: newInterval });

      // 重启本地定时器
      this.startUploadTimer();
      console.log(`已将上传间隔设置为 ${newInterval} 秒`);
    },

    // =========== 后端推送解析 相关 =============
    /** 将后端 JSON 转为 allData 数组 */
    transformParsed(parsed) {
      const arr = [];
      Object.keys(this.fieldMap).forEach(key => {
        if (parsed[key] !== undefined) {
          const { label, unit } = this.fieldMap[key];
          arr.push({
            id: key,
            name: label,
            value: parsed[key],
            unit,
            updateTime: this.formatTime(parsed.time)
          });
        }
      });
      this.allData = arr;
    },

    /** 格式化后端返回的时间戳 */
    formatTime(ts) {
      const y = ts.slice(0,4), m = ts.slice(4,6), d = ts.slice(6,8);
      const H = ts.slice(8,10), M = ts.slice(10,12), S = ts.slice(12,14);
      return `${y}-${m}-${d} ${H}:${M}:${S}`;
    }
  },

  watch: {
    // 监听 uploadInterval 的变化
    uploadInterval(val) {
      this.updateInterval(val);
    }
  },

  mounted() {
    // 页面加载后，先启动一次默认定时器
    this.startUploadTimer();
    // … 其它 mounted 逻辑 …
  },

  beforeDestroy() {
    // 组件销毁前清理定时器
    if (this._uploadTimer) {
      clearInterval(this._uploadTimer);
    }
  },
  mounted() {
    // 使用假数据替代后端数据接收逻辑
    this.allData = [
      { id: 1, name: '溶解氧', value: 11.03, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 2, name: '总磷', value: 0.0291, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 3, name: '叶绿素', value: 11.35, unit: 'µg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 4, name: '地表水水质', value: '2类水', unit: '', updateTime: '2025-06-10 13:20:54' },
      { id: 5, name: '黑臭水体状态', value: '非黑臭水体', unit: '', updateTime: '2025-06-10 13:20:54' },
      { id: 6, name: '富营养状态', value: '贫营养状态', unit: '', updateTime: '2025-06-10 13:20:54' },
      { id: 7, name: '总氮', value: 0.847, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 8, name: '氨氮', value: 0.304, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 9, name: '化学需氧量', value: 7.101, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 10, name: 'PH', value: 8.58, unit: '', updateTime: '2025-06-10 13:20:54' },
      { id: 11, name: '高锰酸钾指数', value: 2.06, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' },
      { id: 12, name: '浊度', value: 43.55, unit: 'FTU', updateTime: '2025-06-10 13:20:54' },
      { id: 13, name: 'fui水色指数', value: 10, unit: '', updateTime: '2025-06-10 13:20:54' },
      { id: 14, name: '透明度', value: 1.2, unit: 'm', updateTime: '2025-06-10 13:20:54' },
      { id: 15, name: '悬浮物浓度', value: 15.8, unit: 'mg/L', updateTime: '2025-06-10 13:20:54' }
    ];
  }
}
</script>

<style scoped>
.dashboard { display: flex; height: 100%; }
.sidebar {
  padding: 20px 10px;
  border-right: 1px solid #ebeef5;
  background: #fff;
}
.filter-group .mt-10 { margin-top: 10px; }
.mt-20 {
  margin-top: 20px;
  padding-left: 10px;
  font-weight: bold;
  color: #333;
}
.metric-menu {
  border-right: none;
  background: #fff;
  max-height: calc(100vh - 180px);
  overflow-y: auto;
}
.toolbar {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 10px;
  background: #fff;
  border-bottom: 1px solid #ebeef5;
}
.view-switcher { display: flex; gap: 8px; }
.switcher-btn {
  width: 36px; height: 36px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; color: #c0c4cc; transition: all 0.2s;
}
.switcher-btn:hover { border-color: #909399; color: #909399; }
.switcher-btn.active { border-color: #409EFF; color: #409EFF; }
.list-icon {
  display: flex; flex-direction: column; justify-content: space-between;
  height: 16px;
}
.list-icon span {
  display: block; width: 20px; height: 2px; background: currentColor;
}
.grid-icon {
  display: grid;
  grid-template-columns: repeat(2, 8px);
  grid-template-rows: repeat(2, 8px);
  gap: 4px;
}
.grid-icon div { width: 8px; height: 8px; background: currentColor; }
.content {
  flex: 1; padding: 20px; background: #f5f7fa; overflow: auto;
}
.card-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
}
.data-card {
  background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  border-radius: 4px; padding: 16px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.data-card .card-header {
  display: flex; justify-content: space-between; align-items: center;
}
.data-card .actions .el-button { margin-left: 5px; }
.data-card .card-body { margin-top: 10px; }
.data-card .value {
  font-size: 24px; font-weight: bold;
}
.data-card .time {
  margin-top: 8px; font-size: 12px; color: #999;
}
.footer {
  padding: 10px; text-align: right; background: #fff;
  border-top: 1px solid #ebeef5;
}
</style>
